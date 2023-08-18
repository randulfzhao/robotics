import cv2
import mediapipe as mp
import pyrealsense2 as rs
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
import warnings
from util.func import *
warnings.filterwarnings('ignore')

"""
基于规则的方法：
如果你的动作有明显的开始和结束特征，你可以定义一些规则来划分动作。
例如，你可以定义当手部的某个关键点达到某个位置时，认为动作开始；
当手部的某个关键点达到另一个位置时，认为动作结束。
"""

# Initialize MediaPipe, RealSense, and check directories
mp_drawing, mp_hands, pipeline = rs_initialize()
check_dirs()

# Initialize video file, DataFrame and related flags
out = None
df_list = []
recording = False
start_detected = False
end_detected = False
end_gesture_detected_time = None

# Start processing each frame for hand detection
with mp_hands.Hands(min_detection_confidence=.7, min_tracking_confidence=.7) as hands:
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
                # Draw the landmarks on the image
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Detect the starting gesture
                if is_start_gesture(hand_landmarks, mp_hands):
                    if not recording and not start_detected:
                        print('Start gesture detected.')
                        filename = get_current_time()
                        out = cv2.VideoWriter(f'vid/{filename}.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640, 480))
                        df_list = []
                        recording = True
                        start_detected = True
                        end_detected = False

                # Detect the ending gesture
                elif is_end_gesture(hand_landmarks, mp_hands):
                    if start_detected and recording and not end_detected:
                        print('End gesture detected.')
                        end_detected = True
                        end_gesture_detected_time = time.time()

                # Stop recording 0.5 seconds after the end gesture is detected
                if end_gesture_detected_time and time.time() - end_gesture_detected_time > 0.5:
                    if len(df_list) > 15:
                        out.release()
                        df = pd.DataFrame(df_list)
                        df.to_csv(f'excel/{filename}.csv', index=False)
                    recording = False
                    start_detected = False
                    end_detected = False
                    end_gesture_detected_time = None

                # If recording is in progress, save the hand landmark data
                if recording:
                    hand_data = []
                    for lm in hand_landmarks.landmark:
                        hand_data.append([lm.x, lm.y, lm.z])
                    df_list.append(hand_data)

        # Write the recorded frames to the video file
        if recording:
            out.write(image)
        # Display the processed frame
        cv2.imshow('MediaPipe Hands', image)
        # Break the loop if the ESC key is pressed
        if cv2.waitKey(5) & 0xFF == 27:
            break

# If recording is in progress when the loop ends, save the data
if recording:
    if len(df_list) > 15:
        out.release()
        df = pd.DataFrame(df_list)
        df.to_csv(f'excel/{filename}.csv', index=False)

# Release resources
pipeline.stop()
cv2.destroyAllWindows()
