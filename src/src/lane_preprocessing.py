#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import rospy 
from std_msgs.msg import Float64, Int64, Bool
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from math import radians
import cv2
import numpy as np

class LanePreprocessing:
    def __init__(self):
        rospy.init_node("lane_preprocessing_node")

        # 서브스크라이버 설정
        rospy.Subscriber("/image_jpeg/compressed", CompressedImage, self.camCB)

        # 변수 초기화
        self.bridge = CvBridge()

        # 차선 감지 관련 변수
        self.img = []  # 카메라 이미지
        self.warped_img = []  # 원근 변환 이미지
        self.grayed_img = []  # 그레이스케일 이미지
        self.bin_img = []  # 이진화 이미지
        self.out_img = []  # 출력 이미지

        rate = rospy.Rate(20)  # 20Hz로 실행
        while not rospy.is_shutdown():
            if len(self.img) != 0:
                self.process_image()
            rate.sleep()

    def process_image(self):
        """
        이미지를 처리하여 차선을 감지 (개선된 버전)
        """
        # 원본 이미지 확인
        cv2.imshow("original", self.img)
        
        # 그레이스케일 먼저 변환
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("1. gray", gray)
        
        # 블러 처리
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        cv2.imshow("2. blurred", blurred)
        
        # 대비 강화
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(blurred)
        cv2.imshow("3. enhanced", enhanced)
        
        # 이진화
        binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                    cv2.THRESH_BINARY_INV, 11, 5)
        cv2.imshow("4. binary", binary)
        
        

        cv2.waitKey(1)


    def camCB(self, msg):
        """
        카메라 이미지 콜백 함수
        압축된 이미지를 opencv 포맷으로 변환
        """
        self.img = self.bridge.compressed_imgmsg_to_cv2(msg)

if __name__ == "__main__":
    try: 
        lane_detection_node = LanePreprocessing()
    except rospy.ROSInterruptException:
        pass