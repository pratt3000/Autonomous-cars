from __future__ import print_function

#Python Headers
import math
import os

# ROS Headers
import rospy

# GEM PACMod Headers
from std_msgs.msg import Header, Bool
from pacmod_msgs.msg import PacmodCmd, PositionWithSpeed
from sensor_msgs.msg import Image
import time

# CV image
from cv_bridge import CvBridge
import cv2
import imutils
import numpy as np

import torch
from utils import detect_streams, detect_static
import cv2
import time


class vars_:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        img_size = 640
        conf_thres = 0.25
        iou_thres = 0.45
        threshold = 70       # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 4  # minimum number of pixels making up a line
        max_line_gap = 70    # maximum gap in pixels between connectable line segments
        lane_boundary_top = 450 # top limit of the box in which we are considering the lane boundaries
        lane_boundary_top_offset = 450
        lane_boundary_bottom_offset = 150
        text_color = (0,0,255)
        implement_half_precision = True
        save_dir = "temp/"
        # source = '0'  # [path to image/video] OR ['0' for camera on PC] 
        source = 'test_data/car.jpeg'
        weights = "model_weights/End-to-end.pth"

class Ego :
    def __init__(self) :

        rospy.init_node('ego', anonymous=True)

        # Config sub/pub
        self.image_sub = rospy.Subscriber("/zed2/zed_node/rgb_raw/image_raw_color", Image, self.callback)
        self.brake_pub = rospy.Publisher("/pacmod/as_rx/brake_cmd", PacmodCmd, queue_size = 1)
        self.accel_pub = rospy.Publisher("/pacmod/as_rx/accel_cmd", PacmodCmd, queue_size = 1)
        self.angle_pub = rospy.Publisher("/pacmod/as_rx/steer_cmd", PositionWithSpeed, queue_size = 1)
        self.angle_msg = PositionWithSpeed()
        self.accel_cmd = PacmodCmd()
        self.enable_pub = rospy.Publisher("/pacmod/as_rx/enable", Bool)
        self.brake_cmd = PacmodCmd()
        self.data_buffer = []

        self.rate = rospy.Rate(10)

        # Ego state
        self.should_emergency_brake = False
        self.cv_bridge = CvBridge()

        # pedestrian detector
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.threshold = 0.5

        # Other config
        self.angular_velocity_limit = 0.4

        # This one should directly go to flashdisk
        self.image_path = None

    # Event loop
    def run(self) :
        # enable control
        self.enable_pub.publish(True)
        while not rospy.is_shutdown() :
            self.rate.sleep()

    def record(self, msg) :
        self.data_buffer.append(msg)
        # self.flush()

    def flush(self) :
        if len(self.data_buffer) > 0 and len(self.data_buffer)%100 == 0 :
            for data in self.data_buffer[-100:] :
                image = self.cv_bridge.imgmsg_to_cv2(data)
                ts = round(time.time() * 1000) # 1/1000 of a second
                fname = str(ts) + ".jpg"
                if self.image_path is None :
                    cv2.imwrite(fname, image)
                else :
                    cv2.imwrite(os.path.join(self.image_path, fname), image)

    """
    input: image matrix
    return value: True if one of the (pedestrian) bboxes have acc>0.8, else returns False
    """
    def pedestrian_exists(self, img) :
        img = imutils.resize(img, width=min(400, img.shape[1]))
        img = img[:,:,0:3]
        (rects, weights) = self.hog.detectMultiScale(
            img.astype(np.uint8), winStride=(4, 4), padding=(8, 8), scale=1.05
        )

        for i in weights:
            if i > self.threshold:
                return True
        return False

    def callback(self, msg) :
        # print("callback", msg)
        self.record(msg)

    # Process current knowns state of the system
    def process(self) :
        if not self.data_buffer:
            # print("qqq")
            return
        

        cv_image = self.cv_bridge.imgmsg_to_cv2(self.data_buffer[-1])

        cv2.imwrite('temp/img.jpeg', cv_image)
        opt = vars_
        opt.source = 'test_data/img.jpeg'

        # you get data
        with torch.no_grad():
            if opt.source != '0':
                # steering angle is in degrees. 
                steering_angle, obstacle_present = detect_static(opt)
                img = cv2.imread('temp/img.jpeg')
                cv2.imshow('image', img)
                cv2.waitKey(1)

        # So that vehicle doesnt turn for small angles

        ''' Angle '''
        self.angle = self.__get_angle(steering_angle)

        ''' Obstacles '''
        if obstacle_present :
            print("obstacle detected")
            self.should_emergency_brake = True
        else :
            print("obstacle not detected")
            self.should_emergency_brake = False
    
            
    def __get_angle(self, angle) :
        if -0.05 < angle < 0.05 :
            return 0
        
        return angle 
    
    def control(self) :

        ''' Angle '''
        self.angle_cmd.angular_position = self.angle
        self.angle_cmd.angular_velocity_limit = self.angular_velocity_limit
        self.angle_pub.publish(self.angle_cmd)

        ''' Obstacle '''
        if self.should_emergency_brake :
            self._apply_brake()
            self.accel_cmd.f64_cmd = 0.0
            self.accel_cmd.enable = True
            self.accel_pub.publish(self.accel_cmd)
        else :
            self._release_brake()
            self.accel_cmd.f64_cmd = 0.32
            self.accel_cmd.enable = True
            self.accel_pub.publish(self.accel_cmd)

    def _apply_brake(self) :
        self.brake_cmd.f64_cmd = 0.6
        self.brake_cmd.enable = True
        self.brake_pub.publish(self.brake_cmd)

    def _release_brake(self) :
        self.brake_cmd.f64_cmd = 0
        self.brake_cmd.enable = True
        self.brake_pub.publish(self.brake_cmd)

        
if __name__ == '__main__':

    ego = Ego()
    ego.run()
    # opt = vars_

    # with torch.no_grad():
    #     if opt.source != '0':
    #         detect_static(opt)
    #     else:
    #         for img, steering_angle, obstacle_in_way in detect_streams(opt):
    #             print(steering_angle)
    #             cv2.imshow('image', img)
    #             cv2.waitKey(1)  # 1 millisecond
        

# Todo:
# 1. Check bboxs and return if one is too big or obj too close (use lidar inferences)
# 2. Warning if going out of drivable region
# 3. Write car translation code
