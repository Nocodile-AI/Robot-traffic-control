#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS node: Simple traffic-aware robot controller (with memory of last detection)

Behaviors:
- "red light"  → STOP (and remember)
- "nocodile"   → STOP (and remember)
- "green light"→ MOVE (and remember)
- Nothing detected:
   • If last was "nocodile" → now safe → MOVE and set last = "nothing"
   • If last was "red light" → stay STOPPED (safety)
   • If last was "green light" or "nothing" → keep previous state

This gives the robot "memory" so it doesn't forget a red light or nocodile until a new clear signal appears.
"""

import os
import rospy
import cv2
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

# Expected class names from your trained model (exact match required)
CLASS_RED_LIGHT    = "red light"
CLASS_GREEN_LIGHT  = "green light"
CLASS_NOCODILE     = "nocodile"

# Control parameters
NORMAL_SPEED = 0.15
STOP_SPEED   = 0.00

# Bounding box area determines distance estimation (ignore when too far away)
TRAFFIC_LIGHT_MIN = 13000
NOCODILE_MIN = 13000

# ────────────────────────────────────────────────
#  MAIN CONTROLLER
# ────────────────────────────────────────────────

class MemoryTrafficController:
    def __init__(self):
        rospy.init_node("memory_traffic_controller", anonymous=False)

        # ─── Model loading ───────────────────────────────
        model_path = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)

        if not os.path.exists(model_path):
            rospy.logerr(f"Model not found: {model_path}")
            rospy.signal_shutdown("Missing model file")
            return

        try:
            self.model = YOLO(model_path)
            rospy.loginfo(f"YOLO model loaded: {model_path}")
            rospy.loginfo(f"Available classes: {list(self.model.names.values())}")
        except Exception as e:
            rospy.logerr(f"Failed to load YOLO model: {e}")
            rospy.signal_shutdown("Model loading failed")
            return

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        rospy.loginfo(f"Using device: {self.device}")

        # ─── ROS interfaces ──────────────────────────────
        self.bridge = CvBridge()

        self.sub_image = rospy.Subscriber(
            "/camera/color/image_raw", Image,
            self.image_callback, queue_size=1
        )

        self.pub_cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=5)
        self.pub_status  = rospy.Publisher("/traffic/status", String, queue_size=5)
        self.pub_viz     = rospy.Publisher("/traffic/image_with_detections", Image, queue_size=1)

        # ─── State with memory ───────────────────────────
        self.current_speed = NORMAL_SPEED
        self.robot_state   = "MOVING"
        self.last_detected = "nothing"          # "red light", "green light", "nocodile", "nothing"

        self.red_detected      = False
        self.green_detected    = False
        self.nocodile_detected = False

        rospy.loginfo("Memory traffic controller initialized (last detection tracking enabled)")

    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"Image conversion failed: {e}")
            return

        # Run detection
        results = self.model(cv_image, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

        self.process_detections(results, cv_image)
        self.make_decision()
        self.publish_control()
        self.publish_status()
        self.publish_visualization(cv_image)

    def process_detections(self, results, image):
        """Reset current-frame flags and draw detections"""
        self.red_detected      = False
        self.green_detected    = False
        self.nocodile_detected = False

        if results.boxes is None:
            return

        for box in results.boxes:
            cls_id = int(box.cls)
            class_name = results.names[cls_id].lower().strip()
            conf = float(box.conf)

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)

            if class_name == CLASS_RED_LIGHT.lower() and area >= TRAFFIC_LIGHT_MIN:
                self.red_detected = True
                self._draw_box(image, x1, y1, x2, y2, "RED LIGHT", conf, (0, 0, 255))

            elif class_name == CLASS_GREEN_LIGHT.lower() and area >= TRAFFIC_LIGHT_MIN:
                self.green_detected = True
                self._draw_box(image, x1, y1, x2, y2, "GREEN LIGHT", conf, (0, 255, 0))

            elif class_name == CLASS_NOCODILE.lower() and area >= NOCODILE_MIN:
                self.nocodile_detected = True
                self._draw_box(image, x1, y1, x2, y2, "NOCODILE", conf, (0, 0, 255))

    def make_decision(self):
        """Decision logic with memory of last detection"""
        if self.red_detected:
            self.last_detected = "red light"
            self.robot_state   = "STOPPED"
            self.current_speed = STOP_SPEED

        elif self.nocodile_detected:
            self.last_detected = "nocodile"
            self.robot_state   = "STOPPED"
            self.current_speed = STOP_SPEED

        elif self.green_detected:
            self.last_detected = "green light"
            self.robot_state   = "MOVING"
            self.current_speed = NORMAL_SPEED

        else:
            # Nothing detected this frame
            if self.last_detected == "nocodile":
                # Special rule: nocodile disappeared → road is clear → MOVE
                self.last_detected = "nothing"
                self.robot_state   = "MOVING"
                self.current_speed = NORMAL_SPEED
            # Otherwise keep previous decision (especially important for red light)
            # e.g. red light disappears → stay STOPPED until green appears

    def publish_control(self):
        cmd = Twist()
        cmd.linear.x  = self.current_speed
        cmd.angular.z = 0.0
        self.pub_cmd_vel.publish(cmd)

    def publish_status(self):
        msg = String()
        msg.data = (
            f"state:{self.robot_state}:speed:{self.current_speed:.2f}:"
            f"red:{self.red_detected}:green:{self.green_detected}:"
            f"nocodile:{self.nocodile_detected}:last:{self.last_detected}"
        )
        self.pub_status.publish(msg)

    def publish_visualization(self, image):
        # Basic status
        cv2.putText(image,
                    f"STATE: {self.robot_state}   SPEED: {self.current_speed:.2f}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Last remembered detection (memory)
        last_color = (0, 255, 255)  # cyan
        cv2.putText(image,
                    f"LAST DETECTED: {self.last_detected.upper()}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, last_color, 2)

        # Current frame warnings
        if self.red_detected or self.nocodile_detected:
            cv2.putText(image, "STOPPING (red / nocodile)", (10, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif self.green_detected:
            cv2.putText(image, "GREEN LIGHT → GO", (10, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        try:
            ros_img = self.bridge.cv2_to_imgmsg(image, "bgr8")
            self.pub_viz.publish(ros_img)
        except CvBridgeError as e:
            rospy.logerr(f"Visualization publish failed: {e}")

        cv2.imshow("Traffic Controller", image)
        cv2.waitKey(1)

    def _draw_box(self, img, x1, y1, x2, y2, label, conf, color):
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.2f}"
        cv2.putText(img, text, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


# ────────────────────────────────────────────────
if __name__ == "__main__" or True:
    try:
        controller = MemoryTrafficController()
        rospy.loginfo("Traffic Controller Started")
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node shutdown")
        while True:pass
    except Exception as e:
        rospy.logerr(f"Fatal error: {e}")
        while True:pass