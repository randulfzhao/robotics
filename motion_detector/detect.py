"""
基于阈值的方法：
你可以定义一些阈值条件，例如当手部关键点的移动速度低于某个阈值时，认为动作已经结束。
这种方法的优点是可以更准确地划分动作，但缺点是可能会受到噪声和其他干扰的影响。
"""

import cv2
import mediapipe as mp
import pyrealsense2 as rs
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

def get_current_time():
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def is_start_gesture(hand_landmarks):
    thumb_tip = [hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x,
                 hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y,
                 hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].z]

    index_finger_tip = [hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y,
                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z]

    distance = np.sqrt(np.sum(np.square(np.subtract(thumb_tip, index_finger_tip))))

    return distance < 0.015

def is_end_gesture(hand_landmarks):
    thumb_tip = [hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x,
                 hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y,
                 hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].z]

    pinky_finger_tip = [hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x,
                        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y,
                        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].z]

    distance = np.sqrt(np.sum(np.square(np.subtract(thumb_tip, pinky_finger_tip))))

    return distance < 0.015


# 初始化 MediaPipe 和 RealSense
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# 检查并创建存储目录
if not os.path.exists('vid'):
    os.makedirs('vid')
if not os.path.exists('excel'):
    os.makedirs('excel')

# 初始化视频文件、DataFrame和相关标志
out = None
df_list = []
recording = False
start_detected = False
end_detected = False
end_gesture_detected_time = None

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        image = np.asanyarray(color_frame.get_data())
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # 检测开始手势
                if is_start_gesture(hand_landmarks):
                    if not recording and not start_detected:
                        print('Start gesture detected.')
                        filename = get_current_time()
                        out = cv2.VideoWriter(f'vid/{filename}.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640, 480))
                        df_list = []
                        recording = True
                        start_detected = True
                        end_detected = False

                # 检测结束手势
                elif is_end_gesture(hand_landmarks):
                    # 只有在检测到开始手势后，我们才处理结束手势
                    if start_detected and recording and not end_detected:
                        print('End gesture detected.')
                        end_detected = True
                        end_gesture_detected_time = time.time()

                # 结束手势检测后0.5秒，停止录制
                if end_gesture_detected_time and time.time() - end_gesture_detected_time > 0.5:
                    if len(df_list) > 15:  # 30fps下的0.5秒为15帧
                        out.release()
                        df = pd.DataFrame(df_list)
                        df.to_csv(f'excel/{filename}.csv', index=False)
                    recording = False
                    start_detected = False
                    end_detected = False
                    end_gesture_detected_time = None

                # 如果正在录制，则保存手部关键点数据
                if recording:
                    hand_data = []
                    for lm in hand_landmarks.landmark:
                        hand_data.append([lm.x, lm.y, lm.z])
                    df_list.append(hand_data)

        # 写入录制的帧
        if recording:
            out.write(image)
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

# 结束循环时，如果仍在录制，保存数据
if recording:
    if len(df_list) > 15: 
        out.release()
        df = pd.DataFrame(df_list)
        df.to_csv(f'excel/{filename}.csv', index=False)

# 释放资源
pipeline.stop()
cv2.destroyAllWindows()