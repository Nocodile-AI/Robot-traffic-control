#!/usr/bin/env python3
# -*- coding: utf-8 -*-
print(".*/import start")
import os
import  sys
import time
import  tty, termios
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import torch
from ultralytics import YOLO
print(".*/import end",YOLO._version)

class TrafficController:
    def __init__(self):
        
		#ä¸äº§çåæ¾ææ
        
        # Initialize ROS node
        rospy.init_node('traffic_controller', anonymous=False)

        
        # Initialize YOLO model
        self.model_path = os.path.join(os.path.dirname(__file__), 'best.pt')
        if not os.path.exists(self.model_path):
            rospy.logerr(f"YOLO model not found at {self.model_path}")
            rospy.logerr("Please download yolo11n.pt and place it in the launch directory")
            return
            
        try:
            self.model = YOLO(self.model_path)
            rospy.loginfo(f"YOLO model loaded successfully from {self.model_path}")
        except Exception as e:
            rospy.logerr(f"Failed to load YOLO model: {e}")
            return
        
        # Check if CUDA is available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        rospy.loginfo(f"Using device: {self.device}")
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Detection parameters
        self.confidence_threshold = 0.5
        self.stable_threshold = 5  # Frames needed for stable detection
        
        # Traffic Light Detection
        self.traffic_light_classes = ['traffic light']
        self.red_light_detected = False
        self.green_light_detected = False
        self.traffic_light_stable_count = 0
        
        # nocodile Detection
        self.nocodile_classes = ['nocodile']
        self.nocodile_detected = False
        self.nocodile_stable_count = 0
        self.zebra_line_detected = False
        
        # Zebra Crossing Detection
        self.zebra_line_classes = ['zebra line']
        self.zebra_line_stable_count = 0
        self.zebra_line_distance_threshold = 150  # pixels (adjust based on camera setup)
        
        # Robot Control
        self.robot_speed = 0.15  # Normal speed
        self.slow_speed = 0.05  # Slow speed for zebra crossing
        self.stop_speed = 0.0   # Stop speed
        
        # Current robot state
        self.current_speed = self.robot_speed
        self.robot_state = "MOVING"  # MOVING, SLOWING, STOPPED
        self.last_stoped_time=0
        self.last_slowed_time=0
        self.last_results=None
        
        # ROS publishers and subscribers
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.traffic_status_pub = rospy.Publisher("/traffic/status", String, queue_size=1)
        self.visualization_pub = rospy.Publisher("/traffic/image_with_detections", Image, queue_size=1)
        
        # Status tracking
        self.detection_history = []
        
        rospy.loginfo("Traffic Controller initialized successfully")

    def image_callback(self, data):
        """Process incoming camera images for traffic management"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"Error converting image: {e}")
            return
        

        print("image_callback",time.time()-self.last_stoped_time)
        if not (time.time() - self.last_stoped_time > 5):
            print("skipped")#//---
            if self.last_results:self.process_detections(self.last_results, cv_image)
            self.publish_visualization_image(cv_image)
            return
        print("image_callback",time.time()-self.last_slowed_time)
        
        #if self.robot_state == "STOPPED":return



        # Run YOLO detection
        results = self.model(cv_image, conf=self.confidence_threshold, verbose=False)
        self.last_results=results
        # Process detections
        self.process_detections(results, cv_image)
        
        # Make traffic decisions
        self.make_traffic_decisions()
        
        # Publish robot control commands
        self.publish_robot_control()
        
        # Publish traffic status
        self.publish_traffic_status()
        
        # Publish visualization image
        self.publish_visualization_image(cv_image)

    def process_detections(self, results, cv_image):
        """Process YOLO detection results for traffic scenarios"""
        # Reset detection flags
        self.red_light_detected = False
        self.green_light_detected = False
        self.nocodile_detected = False
        self.zebra_line_detected = False
        
        for result in results:
            if result.boxes is not None:
                for i, box in enumerate(result.boxes):
                    # Get class name and confidence
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    confidence = float(box.conf[0])
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    width = int(x2 - x1)
                    height = int(y2 - y1)
                    
                    print(class_name)
                    # Traffic Light Detection
                    if class_name in self.traffic_light_classes:
                        self.detect_traffic_light_color(cv_image, int(x1), int(y1), int(x2), int(y2))
                        self.draw_traffic_light_detection(cv_image, int(x1), int(y1), int(x2), int(y2), class_name, confidence)
                    
                    # nocodileren Detection
                    elif class_name in self.nocodile_classes:
                        # Check if nocodile is in zebra crossing area
                        #if self.is_nocodile_on_zebra_line(center_x, center_y, cv_image):
                        self.detect_nocodile(cv_image, int(x1), int(y1), int(x2), int(y2))
                        self.draw_nocodile_detection(cv_image, int(x1), int(y1), int(x2), int(y2), class_name, confidence)
                    
                    # Zebra Crossing Detection
                    elif class_name in self.zebra_line_classes:
                        self.detect_zebra_line(cv_image, int(x1), int(y1), int(x2), int(y2))
                        self.draw_zebra_line_detection(cv_image, int(x1), int(y1), int(x2), int(y2), class_name, confidence)

    def detect_traffic_light(self, image, x1, y1, x2, y2, class_name):
        
        roi = image[y1:y2, x1:x2]
        cv2.imshow("roi",roi)
        if roi.size == 0:
            return
        total_pixels = roi.shape[0] * roi.shape[1]
        print(total_pixels)
        if total_pixels>=13000 and 'green traffic light' in class_name:
            self.green_light_detected = True
        elif total_pixels>=13000 :
            self.red_light_detected = True


    def detect_traffic_light_color(self, image, x1, y1, x2, y2):
        """Detect traffic light color using color analysis"""
        # Extract traffic light region
        roi = image[y1:y2, x1:x2]
        cv2.imshow("roi",roi)
        if roi.size == 0:
            return
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Define color ranges
        # Red color range
        red_lower1 = np.array([0, 50, 50])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 50, 50])
        red_upper2 = np.array([180, 255, 255])
        
        # Green color range
        green_lower = np.array([40, 50, 50])
        green_upper = np.array([80, 255, 255])
        
        # Create masks
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        
        # Count pixels
        red_pixels = cv2.countNonZero(red_mask)
        green_pixels = cv2.countNonZero(green_mask)
        
        # Determine color based on pixel count
        total_pixels = roi.shape[0] * roi.shape[1]
        red_ratio = red_pixels / total_pixels
        green_ratio = green_pixels / total_pixels
        print(total_pixels)
        #if total_pixels>=12000:
            #self.red_light_detected = True

        if red_ratio > 0.1 and total_pixels>=10000:  # Threshold for red detection
           self.red_light_detected = True
        elif green_ratio > 0.1:  # Threshold for green detection
           self.green_light_detected = True


    def detect_nocodile(self, image, x1, y1, x2, y2):
        
        roi = image[y1:y2, x1:x2]
        cv2.imshow("roi",roi)
        if roi.size == 0:
            return
        total_pixels = roi.shape[0] * roi.shape[1]
        print(total_pixels)
        if total_pixels>=13000:
            self.nocodile_detected = True

    def detect_zebra_line(self, image, x1, y1, x2, y2):
        
        roi = image[y1:y2, x1:x2]
        cv2.imshow("roi",roi)
        if roi.size == 0:
            return
        total_pixels = roi.shape[0] * roi.shape[1]
        print(total_pixels)
        if total_pixels>=13000:
            self.zebra_line_detected = True
        

    #def is_nocodile_on_zebra_line(self, center_x, center_y, image):
        """Check if nocodile is on zebra crossing using image analysis"""
        # Look for zebra crossing patterns near the nocodile
        # This is a simplified approach - in practice, you'd use more sophisticated detection
        
        # Check bottom portion of image for zebra crossing patterns
    #    height, width = image.shape[:2]
    #    bottom_region = image[int(height * 0.7):height, :]
        
        # Convert to grayscale
    #    gray = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)
        
        # Look for horizontal lines (zebra crossing stripes)
    #    edges = cv2.Canny(gray, 50, 150)
    #    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=10)
        
    #    if lines is not None:
    #        horizontal_lines = 0
    #        for line in lines:
    #            x1, y1, x2, y2 = line[0]
                # Check if line is roughly horizontal
    #            if abs(y2 - y1) < 10 and abs(x2 - x1) > 50:
    #                horizontal_lines += 1
            
            # If we find multiple horizontal lines, it's likely a zebra crossing
    #        if horizontal_lines >= 3:
    #            return True
        
    #    return False

    def make_traffic_decisions(self):
        """Make traffic control decisions based on detections"""
        # Priority 1: Traffic Light Control
        if self.red_light_detected:
            self.robot_state = "STOPPED"
            self.last_stoped_time = time.time()
            self.current_speed = self.stop_speed
            self.traffic_light_stable_count += 1
        elif self.green_light_detected:
            self.robot_state = "MOVING"
            self.current_speed = self.robot_speed
            self.traffic_light_stable_count += 1
        else:
            self.traffic_light_stable_count = 0
        
        # Priority 2: nocodileren on Zebra Crossing
        if self.nocodile_detected:
            self.robot_state = "STOPPED"
            self.last_stoped_time = time.time()
            self.current_speed = self.stop_speed
            self.nocodile_stable_count += 1
        else:
            self.nocodile_stable_count = 0
        
        # Priority 3: Zebra Crossing Approach (slow down)
        if self.zebra_line_detected :
            self.robot_state = "SLOWING"
            self.current_speed = self.slow_speed
            self.last_slowed_time = time.time()
            self.zebra_line_stable_count += 1
        else:
            self.zebra_line_stable_count = 0
        
        # Default: Normal movement if no traffic conditions
        if (not self.red_light_detected and not self.green_light_detected and 
            not self.nocodile_detected and not self.zebra_line_detected):
            self.robot_state = "MOVING"
            self.current_speed = self.robot_speed

    def publish_robot_control(self):
        """Publish robot movement commands"""
        cmd = Twist()
        
        if self.robot_state == "STOPPED":
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            
        elif self.robot_state == "SLOWING":
            cmd.linear.x = self.slow_speed
            cmd.angular.z = 0.0
        else:  # MOVING
            cmd.linear.x = self.current_speed
            cmd.angular.z = 0.0
        
        self.cmd_vel_pub.publish(cmd)

    def publish_traffic_status(self):
        """Publish traffic status information"""
        status_msg = String()
        status_msg.data = f"state:{self.robot_state}:speed:{self.current_speed}:red_light:{self.red_light_detected}:green_light:{self.green_light_detected}:nocodile:{self.nocodile_detected}:zebra:{self.zebra_line_detected}"
        self.traffic_status_pub.publish(status_msg)

    def draw_traffic_light_detection(self, image, x1, y1, x2, y2, class_name, confidence):
        """Draw traffic light detection with color indication"""
        color = (0, 0, 255) if self.red_light_detected else (0, 255, 0) if self.green_light_detected else (0, 255, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        light_status = "RED" if self.red_light_detected else "GREEN" if self.green_light_detected else "UNKNOWN"
        label = f"{class_name} ({light_status}): {confidence:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def draw_nocodile_detection(self, image, x1, y1, x2, y2, class_name, confidence):
        """Draw nocodile detection on zebra crossing"""
        color = (0, 0, 255)  # Red for danger
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        label = f"NOCODILE: {confidence:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def draw_zebra_line_detection(self, image, x1, y1, x2, y2, class_name, confidence):
        """Draw zebra crossing detection"""
        color = (0, 165, 255)  # Orange for caution
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        label = f"ZEBRA LINE: {confidence:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def publish_visualization_image(self, image):
        """Publish image with traffic detections for visualization"""
        # Add status text to image
        status_text = f"ROBOT STATE: {self.robot_state} | SPEED: {self.current_speed:.2f}"
        cv2.putText(image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add traffic light status
        if self.red_light_detected:
            cv2.putText(image, "RED LIGHT - STOPPING", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        elif self.green_light_detected:
            cv2.putText(image, "GREEN LIGHT - MOVING", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Add nocodile detection status
        if self.nocodile_detected:
            cv2.putText(image, "NOCODILE DETECTED - STOPPING", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Add zebra crossing status
        if self.zebra_line_detected:
            cv2.putText(image, "ZEBRA LINE - SLOWING", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        try:
            ros_image = self.bridge.cv2_to_imgmsg(image, "bgr8")
            self.visualization_pub.publish(ros_image)
        except CvBridgeError as e:
            rospy.logerr(f"Error publishing visualization image: {e}")
        
        # Display image for debugging
        cv2.imshow("Traffic Controller", image)
        cv2.waitKey(1)

if __name__ == '__main__' or True:# !-------------
    try:
        controller = TrafficController()
        rospy.loginfo("Traffic Controller started")
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Traffic Controller stopped")
        while True:pass
    except Exception as e:
        rospy.logerr(f"Traffic Controller error: {e}")
        while True:pass
