#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS node: Memory-based traffic controller using Teachable Machine classification
(Model: TensorFlow Lite .tflite + labels.txt)

Behaviors (same as before):
- "red light"  → STOP (remember)
- "nocodile"   → STOP (remember)
- "green light"→ MOVE (remember)
- Nothing (or low confidence / background class) detected:
   • If last was "nocodile" → now safe → MOVE and set last = "nothing"
   • If last was "red light" → stay STOPPED until green appears
   • Otherwise keep previous state

No bounding boxes — the whole image is classified as ONE class.
"""

import os
import rospy
import cv2
import numpy as np
import tensorflow as tf   # pip install tensorflow  (or tflite-runtime for lighter version)
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from geometry_msgs.msg import Twist

# ────────────────────────────────────────────────
#  CONFIGURATION
# ────────────────────────────────────────────────

TFLITE_MODEL   = "model.tflite"      # your exported Teachable Machine model
LABELS_FILE    = "labels.txt"        # one label per line (exported together)

CONFIDENCE_THRESHOLD = 0.60          # minimum confidence to trust a class (else = "nothing")

# Expected class names (must appear in labels.txt — case-insensitive)
CLASS_RED_LIGHT   = "red light"
CLASS_GREEN_LIGHT = "green light"
CLASS_NOCODILE    = "nocodile"

# Control parameters
NORMAL_SPEED = 0.15
STOP_SPEED   = 0.00

# Input size expected by most Teachable Machine models
INPUT_SIZE = (224, 224)

# ────────────────────────────────────────────────
#  MAIN CONTROLLER
# ────────────────────────────────────────────────

class ClassificationTrafficController:
    def __init__(self):
        rospy.init_node("classification_traffic_controller", anonymous=False)

        # ─── Model & Labels loading ───────────────────────────────
        model_path = os.path.join(os.path.dirname(__file__), TFLITE_MODEL)
        labels_path = os.path.join(os.path.dirname(__file__), LABELS_FILE)

        if not os.path.exists(model_path):
            rospy.logerr(f"Model not found: {model_path}")
            rospy.signal_shutdown("Missing model.tflite")
            return
        if not os.path.exists(labels_path):
            rospy.logerr(f"Labels file not found: {labels_path}")
            rospy.signal_shutdown("Missing labels.txt")
            return

        try:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details  = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            with open(labels_path, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]

            rospy.loginfo(f"Teachable Machine model loaded: {TFLITE_MODEL}")
            rospy.loginfo(f"Classes: {self.labels}")
        except Exception as e:
            rospy.logerr(f"Failed to load model: {e}")
            rospy.signal_shutdown("Model loading failed")
            return

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

        self.current_predicted = "nothing"
        self.current_conf      = 0.0

        rospy.loginfo("Classification traffic controller initialized (Teachable Machine + memory)")

    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"Image conversion failed: {e}")
            return

        self.classify_image(cv_image)
        self.make_decision()
        self.publish_control()
        self.publish_status()
        self.publish_visualization(cv_image)

    def classify_image(self, cv_image):
        """Run Teachable Machine classification on the full image"""
        # Resize to model input size
        resized = cv2.resize(cv_image, INPUT_SIZE)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        input_data = np.expand_dims(normalized, axis=0)

        # Feed to interpreter
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        # Get output
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        max_conf = float(np.max(output))
        class_idx = int(np.argmax(output))

        if max_conf < CONFIDENCE_THRESHOLD:
            predicted_label = "nothing"
        else:
            predicted_label = self.labels[class_idx].strip().lower()

        self.current_predicted = predicted_label
        self.current_conf      = max_conf

        # Set detection flags for decision logic
        self.red_detected      = (predicted_label == CLASS_RED_LIGHT.lower())
        self.green_detected    = (predicted_label == CLASS_GREEN_LIGHT.lower())
        self.nocodile_detected = (predicted_label == CLASS_NOCODILE.lower())

    def make_decision(self):
        """Decision logic with memory of last confident detection"""
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
            # Nothing (or low-confidence) detected this frame
            if self.last_detected == "nocodile":
                # nocodile disappeared → road is now clear
                self.last_detected = "nothing"
                self.robot_state   = "MOVING"
                self.current_speed = NORMAL_SPEED
            # red light disappearance → stay stopped (safety)
            # green or nothing → keep previous decision

    def publish_control(self):
        cmd = Twist()
        cmd.linear.x  = self.current_speed
        cmd.angular.z = 0.0
        self.pub_cmd_vel.publish(cmd)

    def publish_status(self):
        msg = String()
        msg.data = (
            f"state:{self.robot_state}:speed:{self.current_speed:.2f}:"
            f"predicted:{self.current_predicted}:conf:{self.current_conf:.2f}:"
            f"last:{self.last_detected}"
        )
        self.pub_status.publish(msg)

    def publish_visualization(self, image):
        # Big prediction text
        color = (0, 0, 255) if self.red_detected else \
                (0, 255, 0) if self.green_detected else \
                (0, 0, 255) if self.nocodile_detected else (255, 255, 255)

        label = self.current_predicted.upper()
        text = f"PREDICTED: {label} ({self.current_conf:.1%})"
        cv2.putText(image, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)

        # Robot state
        cv2.putText(image,
                    f"STATE: {self.robot_state}   SPEED: {self.current_speed:.2f}",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Memory
        cv2.putText(image,
                    f"LAST REMEMBERED: {self.last_detected.upper()}",
                    (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

        try:
            ros_img = self.bridge.cv2_to_imgmsg(image, "bgr8")
            self.pub_viz.publish(ros_img)
        except CvBridgeError as e:
            rospy.logerr(f"Visualization publish failed: {e}")

        cv2.imshow("Traffic Controller (Classification)", image)
        cv2.waitKey(1)


# ────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        controller = ClassificationTrafficController()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node shutdown")
    except Exception as e:
        rospy.logerr(f"Fatal error: {e}")