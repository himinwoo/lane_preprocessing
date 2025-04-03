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
    
    def multi_scale_retinex(self, image, scales=[15, 80, 250]):
        # BGR을 float 형식으로 변환
        img_float = image.astype(np.float32) / 255.0
        
        # 밝기가 0인 부분 처리 (로그 연산을 위해)
        img_float = np.maximum(img_float, 0.01)
        
        # 채널 분리
        b, g, r = cv2.split(img_float)
        
        # 각 채널에 대해 MSR 적용
        b_msr = self.multi_scale_retinex_channel(b, scales)
        g_msr = self.multi_scale_retinex_channel(g, scales)
        r_msr = self.multi_scale_retinex_channel(r, scales)
        
        # 채널 결합
        img_msr = cv2.merge([b_msr, g_msr, r_msr])
        
        # 정규화 (0-255 범위로)
        img_msr = cv2.normalize(img_msr, None, 0, 255, cv2.NORM_MINMAX)
        img_msr = img_msr.astype(np.uint8)
        
        return img_msr


    def multi_scale_retinex_channel(self, channel, scales):
        """단일 채널에 다중 스케일 Retinex 적용"""
        
        # 로그 변환
        log_channel = np.log10(channel)
        
        # 각 스케일에 대해 가우시안 필터 적용 및 결과 합산
        result = np.zeros_like(channel)
        for scale in scales:
            # 가우시안 블러 (조명 성분 추정)
            blur = cv2.GaussianBlur(channel, (0, 0), scale)
            blur = np.maximum(blur, 0.01)  # 0 방지
            log_blur = np.log10(blur)
            
            # 반사 성분 추정 (원본 - 조명)
            retinex = log_channel - log_blur
            result += retinex
        # 평균
        result /= len(scales)
        return result   

    def remove_shadows(self, image):
        # BGR을 LAB으로 변환
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # L 채널만 분리
        l, a, b = cv2.split(lab)
        
        # CLAHE 적용 (그림자 영역의 대비 개선)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # 채널 병합
        enhanced_lab = cv2.merge([cl, a, b])
        
        # LAB을 BGR로 변환
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return enhanced_bgr

    def process_image(self):
        """
        이미지를 처리하여 차선을 감지
        """

        # 그림자 제거 전처리
        shadow_removed = self.multi_scale_retinex(self.img)
        cv2.imshow("shadow_removed", shadow_removed)
        cv2.imshow("original", self.img)
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