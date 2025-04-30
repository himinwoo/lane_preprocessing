#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import rospy 
from std_msgs.msg import Float64, Int64, Bool
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image
from math import radians
import cv2
import numpy as np

class LanePreprocessing:
    def __init__(self):
        rospy.init_node("lane_preprocessing_node")

        # 서브스크라이버 설정
        rospy.Subscriber("/image_jpeg/compressed", CompressedImage, self.camCB)
        
        # 퍼블리셔 설정 (이진화 결과 이미지 퍼블리시 - 선택사항)
        self.binary_pub = rospy.Publisher('/lane_binary', Image, queue_size=1)
        self.roi_pub = rospy.Publisher('/lane_roi', Image, queue_size=1)  # ROI 결과를 퍼블리시

        # 변수 초기화
        self.bridge = CvBridge()

        # 차선 감지 관련 변수
        self.img = []  # 카메라 이미지
        self.warped_img = []  # 원근 변환 이미지
        self.grayed_img = []  # 그레이스케일 이미지
        self.bin_img = []  # 이진화 이미지
        self.out_img = []  # 출력 이미지
        self.roi_img = []  # ROI 이미지 (추가)
        
        # 이미지 처리 정보 로깅을 위한 카운터
        self.frame_count = 0

        rate = rospy.Rate(20)  # 20Hz로 실행
        while not rospy.is_shutdown():
            if len(self.img) != 0:
                self.process_image()
            rate.sleep()

    def apply_roi(self, image):
        """
        이미지의 아래쪽 절반만 ROI로 추출하는 함수
        """
        height = image.shape[0]
        # 아래쪽 절반만 자르기
        half_height = height // 2
        roi = image[half_height:height, :]
        
        return roi, half_height

    def process_image(self):
        """
        조도 변화에 강건한 이미지 처리 파이프라인 (ROI 적용)
        """
        # 프레임 카운터 증가
        self.frame_count += 1
        
        # 원본 이미지 표시
        cv2.imshow("original", self.img)
        
        # 1. ROI 적용 (이미지 아래 절반만 사용)
        # 원본 이미지의 ROI 추출
        roi_img, half_height = self.apply_roi(self.img)
        # ROI 결과 표시
        cv2.imshow("1. ROI", roi_img)
        self.roi_img = roi_img  # ROI 이미지 저장
        
        # 2. 그레이스케일 변환 (ROI 이미지에 적용)
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("2. gray", gray)
        
        # 3. 블러 처리 (노이즈 제거)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 4. CLAHE로 대비 향상 (조도 변화 완화)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        cv2.imshow("3. enhanced", enhanced)
        
        # 5. 이미지 통계 분석
        mean_val = np.mean(enhanced)
        std_val = np.std(enhanced)
        
        # 6. 이미지 특성에 따라 최적의 이진화 방법 선택
        if std_val < 20:  # 조도가 균일한 경우
            # 오츠의 알고리즘 적용
            _, binary_otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary = binary_otsu  # 최종 이진화 이미지로 오츠 결과 사용
            method = "Otsu"
            
        elif std_val < 50:  # 조도가 불균일하지만 변화가 과도하지 않은 경우
            # 중간 크기의 블록으로 적응형 이진화 적용
            binary_adaptive = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 21, 7
            )
            binary = binary_adaptive  # 최종 이진화 이미지로 적응형 결과 사용
            method = "Adaptive (block=21, C=7)"
            
        else:  # 조도 변화가 심한 경우
            # 작은 블록 크기와 더 작은 C 값 사용
            binary_adaptive = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 3
            )
            
            # 노이즈 제거를 위한 모폴로지 연산 추가
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary_adaptive, cv2.MORPH_OPEN, kernel)
            method = "Adaptive (block=11, C=3) + Morphology"
        
        # 이진화 결과 표시
        cv2.imshow("4. binary", binary)
        
        # 7. 결과 저장 및 퍼블리시
        self.bin_img = binary
        
        # 8. 결과를 원본 이미지 크기로 매핑 (선택사항)
        # 전체 이미지 크기의 이진화 결과를 생성하고 싶은 경우
        full_binary = np.zeros((self.img.shape[0], self.img.shape[1]), dtype=np.uint8)
        # ROI 결과를 아래쪽 절반에 복사
        full_binary[half_height:, :] = binary
        
        # 전체 이미지에 매핑된 결과 표시
        cv2.imshow("5. full binary", full_binary)
        
        # 10프레임마다 처리 방법 로깅 (콘솔 출력 줄이기)
        if self.frame_count % 10 == 0:
            rospy.loginfo(f"이진화 방법: {method}, 평균 밝기: {mean_val:.1f}, 표준편차: {std_val:.1f}")
            
            # 퍼블리시 (선택사항, 다른 노드에서 이진화 결과를 사용하고 싶은 경우)
            try:
                # ROI 이미지 퍼블리시
                roi_msg = self.bridge.cv2_to_imgmsg(roi_img, encoding="bgr8")
                self.roi_pub.publish(roi_msg)
                
                # 이진화 결과 퍼블리시 (ROI 영역만)
                binary_msg = self.bridge.cv2_to_imgmsg(binary, encoding="mono8")
                self.binary_pub.publish(binary_msg)
            except Exception as e:
                rospy.logwarn(f"이미지 퍼블리시 중 오류: {e}")
        
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