#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS node: Memory-based traffic controller using frozen TensorFlow .pb model

Behaviors (same as before):
- "red light"  → STOP (remember)
- "nocodile"   → STOP (remember)
- "green light"→ MOVE (remember)
- Nothing/low confidence:
  • If last was "nocodile" → safe → MOVE + last = "nothing"
  • If last was "red light" → stay STOPPED until green
"""

import os
import rospy
import cv2
import numpy as np
import tensorflow as tf
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from geometry_msgs.msg import Twist

# ────────────────────────────────────────────────
#  CONFIGURATION – YOU MUST UPDATE THESE
# ────────────────────────────────────────────────

PB_MODEL_FILE = "model.pb"           # your downloaded .pb file

# !!! CHANGE THESE to match your model's tensor names !!!
INPUT_TENSOR_NAME  = "input_1:0"     # e.g. "Placeholder:0", "images:0", "input:0"
OUTPUT_TENSOR_NAME = "Identity:0"    # e.g. "dense/Softmax:0", "predictions/Softmax:0"

CONFIDENCE_THRESHOLD = 0.60          # min confidence to accept a class (else "nothing")

# Expected class names (must match model's output order / argmax mapping)
# Adjust index ↔ label mapping below if your model has different order
CLASS_NAMES = [
    "nothing",       # index 0 - background / no relevant object
    "red light",     # index 1
    "green light",   # index 2
    "nocodile",      # index 3
    # add more if your model has more classes
]

# Control parameters
NORMAL_SPEED = 0.15
STOP_SPEED   = 0.00

# Typical Teachable-Machine-like input size
INPUT_SIZE = (224, 224)

# ────────────────────────────────────────────────
#  MAIN CONTROLLER
# ────────────────────────────────────────────────

class PbTrafficController:
    def __init__(self):
        rospy.init_node("pb_traffic_controller", anonymous=False)

        # ─── Load frozen .pb model ───────────────────────────────
        model_path = os.path.join(os.path.dirname(__file__), PB_MODEL_FILE)

        if not os.path.exists(model_path):
            rospy.logerr(f"Model not found: {model_path}")
            rospy.signal_shutdown("Missing .pb file")
            return

        try:
            self.graph = tf.Graph()
            self.sess = tf.compat.v1.Session(graph=self.graph)

            with tf.compat.v1.gfile.GFile(model_path, "rb") as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name="")

            # Get handles to input/output tensors
            self.input_tensor  = self.graph.get_tensor_by_name(INPUT_TENSOR_NAME)
            self.output_tensor = self.graph.get_tensor_by_name(OUTPUT_TENSOR_NAME)

            rospy.loginfo(f"Frozen .pb model loaded: {PB_MODEL_FILE}")
            rospy.loginfo(f"Input tensor:  {INPUT_TENSOR_NAME}")
            rospy.loginfo(f"Output tensor: {OUTPUT_TENSOR_NAME}")
        except Exception as e:
            rospy.logerr(f"Failed to load .pb model: {e}")
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
        self.last_detected = "nothing"

        self.current_predicted = "nothing"
        self.current_conf      = 0.0

        rospy.loginfo("Frozen .pb traffic controller initialized (with memory)")

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
        """Run inference with frozen .pb graph"""
        # Preprocess: resize → RGB → normalize [0,1]
        resized = cv2.resize(cv_image, INPUT_SIZE)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        input_data = np.expand_dims(normalized, axis=0)

        # Run session
        output = self.sess.run(
            self.output_tensor,
            feed_dict={self.input_tensor: input_data}
        )[0]  # shape (num_classes,)

        max_conf = float(np.max(output))
        class_idx = int(np.argmax(output))

        if max_conf < CONFIDENCE_THRESHOLD:
            predicted_label = "nothing"
        else:
            try:
                predicted_label = CLASS_NAMES[class_idx].lower().strip()
            except IndexError:
                predicted_label = "unknown"
                rospy.logwarn(f"Class index {class_idx} out of range")

        self.current_predicted = predicted_label
        self.current_conf      = max_conf

        # Detection flags for decision
        self.red_detected      = (predicted_label == "red light")
        self.green_detected    = (predicted_label == "green light")
        self.nocodile_detected = (predicted_label == "nocodile")

    def make_decision(self):
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
            # Nothing / low confidence this frame
            if self.last_detected == "nocodile":
                self.last_detected = "nothing"
                self.robot_state   = "MOVING"
                self.current_speed = NORMAL_SPEED
            # red disappearance → stay stopped (safety)

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
        color = (0, 0, 255) if self.red_detected else \
                (0, 255, 0) if self.green_detected else \
                (0, 0, 255) if self.nocodile_detected else (255, 255, 255)

        label = self.current_predicted.upper()
        text = f"PREDICTED: {label} ({self.current_conf:.1%})"
        cv2.putText(image, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)

        cv2.putText(image,
                    f"STATE: {self.robot_state}   SPEED: {self.current_speed:.2f}",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.putText(image,
                    f"LAST: {self.last_detected.upper()}",
                    (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

        try:
            ros_img = self.bridge.cv2_to_imgmsg(image, "bgr8")
            self.pub_viz.publish(ros_img)
        except CvBridgeError as e:
            rospy.logerr(f"Visualization publish failed: {e}")

        cv2.imshow("Traffic Controller (.pb)", image)
        cv2.waitKey(1)


# ────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        controller = PbTrafficController()
        rospy.spin()

    except rospy.ROSInterruptException:
        rospy.loginfo("Node shutdown")
        while True:pass
    except Exception as e:
        rospy.logerr(f"Fatal error: {e}")
        while True:pass
    finally:
        # Clean up session
        if hasattr(controller, 'sess'):
            controller.sess.close()