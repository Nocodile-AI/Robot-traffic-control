#!/usr/bin/env python3
"""
ROS node: Memory-based traffic controller using a Teachable Machine Keras model.

Behaviors (identical to traffic_control.py):
  - "red light"   -> STOP (remember)
  - "nocodile"    -> STOP (remember)
  - "green light" -> MOVE (remember)
  - Nothing / low confidence:
      * If last was "nocodile" -> safe -> MOVE and reset last = "nothing"
      * If last was "red light" -> stay STOPPED until a green light is seen
      * Otherwise keep the previous state

Model:
  keras_model.h5 exported from Google Teachable Machine (Keras format).
  Normalization expected by the model: (pixel / 127.5) - 1  (range [-1, 1]).
"""

import os

import cv2
import numpy as np
import rospy
import tensorflow as tf
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import String

KERAS_MODEL_FILE = "keras_model.h5"

CONFIDENCE_THRESHOLD = 0.50

CLASS_RED_LIGHT = "red light"
CLASS_GREEN_LIGHT = "green light"
CLASS_NOCODILE = "nocodile"

# Must match the label order that the model was trained with.
# Google Teachable Machine exports a labels.txt alongside the model; update
# this list to match that file exactly if your training used different classes
# or a different order.
CLASS_NAMES = [
    "nothing",
    "red light",
    "green light",
    "nocodile",
]

NORMAL_SPEED = 0.15
STOP_SPEED = 0.00

INPUT_SIZE = (224, 224)


class KerasTrafficController:
    def __init__(self):
        rospy.init_node("keras_traffic_controller", anonymous=False)

        model_path = os.path.join(os.path.dirname(__file__), KERAS_MODEL_FILE)

        if not os.path.exists(model_path):
            rospy.logerr(f"Model not found: {model_path}")
            rospy.signal_shutdown("Missing model file")
            return

        try:
            self.model = tf.keras.models.load_model(model_path, compile=False)
            rospy.loginfo(f"Keras model loaded: {model_path}")
        except Exception as e:
            rospy.logerr(f"Failed to load Keras model: {e}")
            rospy.signal_shutdown("Model loading failed")
            return

        self.bridge = CvBridge()

        self.sub_image = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback, queue_size=1)

        self.pub_cmd_vel = rospy.Publisher("/cmd_vel", Twist, queue_size=5)
        self.pub_status = rospy.Publisher("/traffic/status", String, queue_size=5)
        self.pub_viz = rospy.Publisher("/traffic/image_with_detections", Image, queue_size=1)

        self.current_speed = NORMAL_SPEED
        self.robot_state = "MOVING"
        self.last_detected = "nothing"

        self.red_detected = False
        self.green_detected = False
        self.nocodile_detected = False

        self.current_predicted = "nothing"
        self.current_conf = 0.0

        rospy.loginfo("Keras traffic controller initialized (last detection tracking enabled)")

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
        """Run Keras inference and update per-frame detection flags."""
        self.red_detected = False
        self.green_detected = False
        self.nocodile_detected = False

        resized = cv2.resize(cv_image, INPUT_SIZE)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = (rgb.astype(np.float32) / 127.5) - 1
        input_data = np.expand_dims(normalized, axis=0)

        output = self.model.predict(input_data, verbose=0)[0]

        max_conf = float(np.max(output))
        class_idx = int(np.argmax(output))

        if max_conf < CONFIDENCE_THRESHOLD:
            predicted_label = "nothing"
        else:
            try:
                predicted_label = CLASS_NAMES[class_idx].lower().strip()
            except IndexError:
                predicted_label = "unknown"
                rospy.logwarn(f"Class index {class_idx} out of range for CLASS_NAMES (len={len(CLASS_NAMES)})")

        self.current_predicted = predicted_label
        self.current_conf = max_conf

        self.red_detected = predicted_label == CLASS_RED_LIGHT
        self.green_detected = predicted_label == CLASS_GREEN_LIGHT
        self.nocodile_detected = predicted_label == CLASS_NOCODILE

    def make_decision(self):
        """Decision logic with memory of last detection (copied from traffic_control.py)."""
        if self.red_detected:
            self.last_detected = "red light"
            self.robot_state = "STOPPED"
            self.current_speed = STOP_SPEED

        elif self.nocodile_detected:
            self.last_detected = "nocodile"
            self.robot_state = "STOPPED"
            self.current_speed = STOP_SPEED

        elif self.green_detected:
            self.last_detected = "green light"
            self.robot_state = "MOVING"
            self.current_speed = NORMAL_SPEED

        else:
            # Nothing detected this frame.
            if self.last_detected == "nocodile":
                # Special rule: nocodile disappeared -> road is clear -> MOVE.
                self.last_detected = "nothing"
                self.robot_state = "MOVING"
                self.current_speed = NORMAL_SPEED
            # Otherwise keep previous decision (especially important for red light:
            # if the red light disappears, stay STOPPED until green appears).

    def publish_control(self):
        cmd = Twist()
        cmd.linear.x = self.current_speed
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
        cv2.putText(
            image,
            f"STATE: {self.robot_state}   SPEED: {self.current_speed:.2f}",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            image,
            f"LAST DETECTED: {self.last_detected.upper()}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
        )

        if self.red_detected or self.nocodile_detected:
            cv2.putText(
                image,
                "STOPPING (red / nocodile)",
                (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
        elif self.green_detected:
            cv2.putText(
                image,
                "GREEN LIGHT -> GO",
                (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        cv2.putText(
            image,
            f"PREDICTED: {self.current_predicted.upper()} ({self.current_conf:.1%})",
            (10, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            2,
        )

        try:
            ros_img = self.bridge.cv2_to_imgmsg(image, "bgr8")
            self.pub_viz.publish(ros_img)
        except CvBridgeError as e:
            rospy.logerr(f"Visualization publish failed: {e}")

        cv2.imshow("Traffic Controller (Keras)", image)
        cv2.waitKey(1)


if __name__ == "__main__":
    try:
        controller = KerasTrafficController()
        rospy.loginfo("Traffic Controller Started")
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node shutdown")
    except Exception as e:
        rospy.logerr(f"Fatal error: {e}")