#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS node: Simple traffic-aware robot controller
Stops on: red light OR nocodile
Goes on: green light (or nothing dangerous detected)
"""

import os
import time
import rospy
import cv2
import numpy as np
import torch
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from ultralytics import YOLO

# ────────────────────────────────────────────────
#  CONFIGURATION
# ────────────────────────────────────────────────

MODEL_FILENAME = "best.pt"

CONFIDENCE_THRESHOLD = 0.50

# Detection classes (must match your trained model)
CLASS_TRAFFIC_LIGHT = "traffic light"
CLASS_NOCODILE      = "nocodile"

# Control parameters
NORMAL_SPEED = 0.15
STOP_SPEED   = 0.00

MIN_ROI_PIXELS        = 10000   # minimum area to trust a detection
MIN_PIXEL_RATIO_COLOR = 0.10    # fraction of red/green pixels needed in traffic light ROI

# How long to stay stopped after red/nocodile disappears (safety buffer)
STOP_COOLDOWN_SECONDS = 4.0

# ────────────────────────────────────────────────
#  MAIN CONTROLLER CLASS
# ────────────────────────────────────────────────

class SimpleTrafficController:
    def __init__(self):
        rospy.init_node("simple_traffic_controller", anonymous=False)

        # ─── Model Loading ───────────────────────────────
        model_path = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)

        if not os.path.exists(model_path):
            rospy.logerr(f"Model not found: {model_path}")
            rospy.signal_shutdown("Missing model file")
            return

        try:
            self.model = YOLO(model_path)
            rospy.loginfo(f"YOLO model loaded: {model_path}")
        except Exception as e:
            rospy.logerr(f"Failed to load YOLO model: {e}")
            rospy.signal_shutdown("Model loading failed")
            return

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        rospy.loginfo(f"Using device: {self.device}")

        # ─── ROS Interface ───────────────────────────────
        self.bridge = CvBridge()

        self.sub_image = rospy.Subscriber(
            "/camera/color/image_raw", Image,
            self.image_callback, queue_size=1
        )

        self.pub_cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=5)
        self.pub_status  = rospy.Publisher("/traffic/status", String, queue_size=5)
        self.pub_viz     = rospy.Publisher("/traffic/image_with_detections", Image, queue_size=2)

        # ─── State ───────────────────────────────────────
        self.last_danger_time = 0.0     # timestamp of last red light or nocodile

        self.red_detected     = False
        self.green_detected   = False
        self.nocodile_detected = False

        self.robot_state      = "MOVING"
        self.current_speed    = NORMAL_SPEED

        rospy.loginfo("Simple traffic controller initialized")

    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"Image conversion failed: {e}")
            return

        now = time.time()

        # Skip heavy detection during short cooldown after danger disappears
        if now - self.last_danger_time < STOP_COOLDOWN_SECONDS:
            self.publish_visualization(cv_image)
            self.publish_control_command()
            self.publish_status()
            return

        # Run YOLO
        results = self.model(cv_image, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

        self.process_detections(results, cv_image)
        self.make_decision(now)
        self.publish_control_command()
        self.publish_status()
        self.publish_visualization(cv_image)

    def process_detections(self, results, image):
        """Reset flags and evaluate every detection"""
        self.red_detected      = False
        self.green_detected    = False
        self.nocodile_detected = False

        if results.boxes is None:
            return

        for box in results.boxes:
            cls_id = int(box.cls)
            class_name = results.names[cls_id]
            conf = float(box.conf)

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if class_name == CLASS_TRAFFIC_LIGHT:
                self._analyze_traffic_light_color(image, x1, y1, x2, y2)
                self._draw_box(image, x1, y1, x2, y2, "Traffic Light", conf, self._get_light_color())

            elif class_name == CLASS_NOCODILE:
                self._check_nocodile_size(x1, y1, x2, y2)
                self._draw_box(image, x1, y1, x2, y2, "NOCODILE", conf, (0, 0, 255))

    def _analyze_traffic_light_color(self, img, x1, y1, x2, y2):
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            return

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Red ranges
        red_mask1 = cv2.inRange(hsv, np.array([0,   50, 50]), np.array([10,  255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        # Green range
        green_mask = cv2.inRange(hsv, np.array([40, 50, 50]), np.array([80, 255, 255]))

        red_count   = cv2.countNonZero(red_mask)
        green_count = cv2.countNonZero(green_mask)
        total       = roi.shape[0] * roi.shape[1]

        if total < MIN_ROI_PIXELS:
            return

        red_ratio   = red_count   / total
        green_ratio = green_count / total

        if red_ratio > MIN_PIXEL_RATIO_COLOR:
            self.red_detected = True
        elif green_ratio > MIN_PIXEL_RATIO_COLOR:
            self.green_detected = True

    def _check_nocodile_size(self, x1, y1, x2, y2):
        area = (x2 - x1) * (y2 - y1)
        if area >= MIN_ROI_PIXELS:
            self.nocodile_detected = True

    def make_decision(self, now: float):
        danger = self.red_d