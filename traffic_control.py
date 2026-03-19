#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS node: Intelligent traffic-aware robot controller
Detects: traffic lights (red/green), "nocodile" (pedestrian?), zebra crossings
Decides: stop / slow down / go
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
CLASS_NOCODILE     = "nocodile"
CLASS_ZEBRA        = "zebra line"

# Control parameters
NORMAL_SPEED   = 0.15
SLOW_SPEED     = 0.05
STOP_SPEED     = 0.00

MIN_ROI_PIXELS        = 10000    # minimum area to trust detection
MIN_PIXEL_RATIO_COLOR = 0.10     # for traffic light color detection

# Time-based cooldowns (seconds)
MIN_STOP_DURATION     = 5.0
MIN_SLOW_DURATION     = 3.0      # how long we stay in slow mode after zebra disappears

# ────────────────────────────────────────────────
#  MAIN CONTROLLER CLASS
# ────────────────────────────────────────────────

class TrafficAwareController:
    def __init__(self):
        rospy.init_node("traffic_aware_controller", anonymous=False)

        # ─── Model ───────────────────────────────────────
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

        # ─── ROS ─────────────────────────────────────────
        self.bridge = CvBridge()

        self.sub_image = rospy.Subscriber(
            "/camera/color/image_raw", Image,
            self.image_callback, queue_size=1
        )

        self.pub_cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=5)
        self.pub_status  = rospy.Publisher("/traffic/status", String, queue_size=5)
        self.pub_viz     = rospy.Publisher("/traffic/image_with_detections", Image, queue_size=1)

        # ─── State ───────────────────────────────────────
        self.last_stop_time  = 0
        self.last_slow_time  = 0
        self.last_results    = None

        self.robot_state     = "MOVING"
        self.current_speed   = NORMAL_SPEED

        rospy.loginfo("Traffic-aware controller initialized")

    # ────────────────────────────────────────────────
    #  MAIN IMAGE CALLBACK
    # ────────────────────────────────────────────────

    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"Image conversion failed: {e}")
            return

        now = time.time()

        # Skip heavy processing shortly after stopping (cooldown)
        if now - self.last_stop_time < MIN_STOP_DURATION:
            if self.last_results:
                self.process_detections(self.last_results, cv_image)
            self.publish_visualization(cv_image)
            return

        # Run detection
        results = self.model(cv_image, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
        self.last_results = results

        self.process_detections(results, cv_image)
        self.make_decision()
        self.publish_control_command()
        self.publish_status()
        self.publish_visualization(cv_image)

    # ────────────────────────────────────────────────
    #  DETECTION PROCESSING
    # ────────────────────────────────────────────────

    def process_detections(self, results, image):
        """Reset flags and process every detected object"""
        # Reset all flags at the beginning of each frame
        self.red_light_detected   = False
        self.green_light_detected = False
        self.nocodile_detected    = False
        self.zebra_detected       = False

        if results.boxes is None:
            return

        for box in results.boxes:
            cls_id = int(box.cls)
            class_name = results.names[cls_id]
            conf = float(box.conf)

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if class_name == CLASS_TRAFFIC_LIGHT:
                self._analyze_traffic_light_color(image, x1, y1, x2, y2)
                self._draw_box(image, x1, y1, x2, y2, class_name, conf, self._get_light_color())

            elif class_name == CLASS_NOCODILE:
                self._check_nocodile_size(x1, y1, x2, y2)
                self._draw_box(image, x1, y1, x2, y2, "NOCODILE", conf, (0, 0, 255))

            elif class_name == CLASS_ZEBRA:
                self._check_zebra_size(x1, y1, x2, y2)
                self._draw_box(image, x1, y1, x2, y2, "ZEBRA", conf, (0, 165, 255))

    def _analyze_traffic_light_color(self, img, x1, y1, x2, y2):
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            return

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Red (two ranges)
        red_mask1 = cv2.inRange(hsv, np.array([0, 50, 50]),   np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        # Green
        green_mask = cv2.inRange(hsv, np.array([40, 50, 50]), np.array([80, 255, 255]))

        red_count   = cv2.countNonZero(red_mask)
        green_count = cv2.countNonZero(green_mask)
        total       = roi.shape[0] * roi.shape[1]

        if total < MIN_ROI_PIXELS:
            return

        if red_count / total > MIN_PIXEL_RATIO_COLOR:
            self.red_light_detected = True
        elif green_count / total > MIN_PIXEL_RATIO_COLOR:
            self.green_light_detected = True

    def _check_nocodile_size(self, x1, y1, x2, y2):
        area = (x2 - x1) * (y2 - y1)
        if area >= MIN_ROI_PIXELS:
            self.nocodile_detected = True

    def _check_zebra_size(self, x1, y1, x2, y2):
        area = (x2 - x1) * (y2 - y1)
        if area >= MIN_ROI_PIXELS:
            self.zebra_detected = True

    # ────────────────────────────────────────────────
    #  DECISION LOGIC (priority order)
    # ────────────────────────────────────────────────

    def make_decision(self):
        now = time.time()

        if self.red_light_detected or self.nocodile_detected:
            self.robot_state = "STOPPED"
            self.current_speed = STOP_SPEED
            self.last_stop_time = now

        elif self.zebra_detected or (now - self.last_stop_time < MIN_SLOW_DURATION):
            self.robot_state = "SLOWING"
            self.current_speed = SLOW_SPEED
            self.last_slow_time = now

        else:
            self.robot_state = "MOVING"
            self.current_speed = NORMAL_SPEED

    # ────────────────────────────────────────────────
    #  PUBLISHERS
    # ────────────────────────────────────────────────

    def publish_control_command(self):
        cmd = Twist()
        cmd.linear.x = self.current_speed
        cmd.angular.z = 0.0
        self.pub_cmd_vel.publish(cmd)

    def publish_status(self):
        msg = String()
        msg.data = (
            f"state:{self.robot_state}:speed:{self.current_speed:.2f}:"
            f"red:{self.red_light_detected}:green:{self.green_light_detected}:"
            f"nocodile:{self.nocodile_detected}:zebra:{self.zebra_detected}"
        )
        self.pub_status.publish(msg)

    def publish_visualization(self, image):
        # Add overlay text
        cv2.putText(image, f"STATE: {self.robot_state}  SPEED: {self.current_speed:.2f}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if self.red_light_detected:
            cv2.putText(image, "RED LIGHT → STOP", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif self.green_light_detected:
            cv2.putText(image, "GREEN LIGHT → GO", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if self.nocodile_detected:
            cv2.putText(image, "NOCODILE DETECTED → STOP", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if self.zebra_detected:
            cv2.putText(image, "ZEBRA CROSSING → SLOW", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        try:
            ros_img = self.bridge.cv2_to_imgmsg(image, "bgr8")
            self.pub_viz.publish(ros_img)
        except CvBridgeError as e:
            rospy.logerr(f"Visualization publish failed: {e}")

        cv2.imshow("Traffic Controller", image)
        cv2.waitKey(1)

    # ────────────────────────────────────────────────
    #  DRAWING HELPERS
    # ────────────────────────────────────────────────

    def _draw_box(self, img, x1, y1, x2, y2, label, conf, color):
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.2f}"
        cv2.putText(img, text, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def _get_light_color(self):
        if self.red_light_detected:
            return (0, 0, 255)
        if self.green_light_detected:
            return (0, 255, 0)
        return (0, 255, 255)  # yellow = unknown

# ────────────────────────────────────────────────
#  ENTRY POINT
# ────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        controller = TrafficAwareController()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node shutdown")
    except Exception as e:
        rospy.logerr(f"Fatal error: {e}")