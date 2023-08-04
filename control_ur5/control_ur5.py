# usable demo backup
import cv2
import numpy as np
import mediapipe as mp
from pyrealsense2 import pyrealsense2 as rs
from controller import Robot
from ikpy.chain import Chain

# for testing performance
import pandas as pd
original = list()
after = list()

# Mediapipe相关设置
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)  # 初始化手部检测模型

# 设置realsense相机
pipe = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipe.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# 获取相机内参
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()


# 定义一个模拟的UR5机器人类
class myUR5(Robot):
    def __init__(self):
        super().__init__()
        self.timestep = int(self.getBasicTimeStep())

        # 获取机器人的电机
        self.motors = [
            self.getDevice("shoulder_pan_joint"),
            self.getDevice("shoulder_lift_joint"),
            self.getDevice("elbow_joint"),
            self.getDevice("wrist_1_joint"),
            self.getDevice("wrist_2_joint"),
            self.getDevice("wrist_3_joint")
        ]

        # 定义活动链接的遮罩
        active_links_mask = [False, True, True, True, True, True, False, False]

        # 使用ikpy从URDF文件中定义UR5机器人的运动链
        self.chain = Chain.from_urdf_file("ur5e.urdf", active_links_mask=active_links_mask)

        # 获取每个关节的位置传感器
        self.position_sensors = [
            self.getDevice("shoulder_pan_joint_sensor"),
            self.getDevice("shoulder_lift_joint_sensor"),
            self.getDevice("elbow_joint_sensor"),
            self.getDevice("wrist_1_joint_sensor"),
            self.getDevice("wrist_2_joint_sensor"),
            self.getDevice("wrist_3_joint_sensor")
        ]

        # 启用位置传感器
        for sensor in self.position_sensors:
            sensor.enable(self.timestep)

    def set_pos(self, positions):
        # 将每个关节的位置设置为目标位置
        for i, pos in enumerate(positions):
            self.motors[i].setPosition(pos)
        # 步进模拟以应用新的位置
        self.step(self.timestep)

    def get_pos(self):
        # 获取每个关节的当前位置
        current_positions = [sensor.getValue() for sensor in self.position_sensors]
        return current_positions

    def set_joint_positions(self, positions):
        # 将每个关节的位置设置为目标位置
        for i, pos in enumerate(positions):
            self.motors[i].setPosition(pos)
        self.step(self.timestep)  # 步进模拟以应用新的位置

    def inverse_kinematics(self, target_position):
        # 计算目标位置的逆运动学
        joint_positions = self.chain.inverse_kinematics(target_position)
        return joint_positions[1:7]  # 忽略第一个和最后一个关节，因为它们是固定的

def rescale(value, old_min, old_max, new_min=-1.1, new_max=1.1):
    return ((value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min

if __name__ == "__main__":
    arm = myUR5()  # 初始化机器人

    try:
        while True:  # 当RGB-D摄像头打开时
            frames = pipe.wait_for_frames()  # 读取帧
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            # 在处理前，将BGR图像转换为RGB。
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            
            results = hands.process(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

            if results.multi_hand_landmarks:  # 如果找到了手部标记
                for hand_landmarks in results.multi_hand_landmarks:
                    for id, lm in enumerate(hand_landmarks.landmark):
                        if id == mp_hands.HandLandmark.WRIST.value:  # 如果是手腕
                            wrist_x = int(lm.x * color_image.shape[1])
                            wrist_y = int(lm.y * color_image.shape[0])
                            wrist_x = max(0, min(719, wrist_x))  # 假设宽度也是720，需要根据实际情况调整
                            wrist_y = max(0, min(719, wrist_y))
                            wrist_z = depth_image[wrist_y, wrist_x].astype(float)
                            if wrist_z==0:
                                continue
                            
                            original_position = [wrist_x,wrist_y,wrist_z]
                            original.append(original_position)

                            # 将像素坐标转换为真实世界坐标，后续需要调节相应参数
                            x_rescaled = rescale(wrist_x, 109, 588)
                            y_rescaled = rescale(wrist_y, 36, 476)
                            z_rescaled = rescale(wrist_z, 500, 1200)


                            # 转换坐标系，并可能需要按比例缩放
                            scale_factor = 1  # 需要根据实际情况调整
                            target_position = [x_rescaled,z_rescaled,-y_rescaled]
                            after.append(target_position)
                            joint_positions = arm.inverse_kinematics(target_position)
                            arm.set_joint_positions(joint_positions)

            # 在图像上绘制手部注释
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(color_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 显示图像
            cv2.imshow('MediaPipe Hands', color_image)
            cv2.imshow('Depth Image', depth_colormap)
            if cv2.waitKey(5) & 0xFF == 27:  # 如果按下ESC，则退出

                outputs = pd.DataFrame({'original': original, 'after': after})
                outputs.to_csv("positions.csv",index=False)
                break

    finally:
        pipe.stop()  # 关闭RGB-D摄像头
        cv2.destroyAllWindows()  # 关闭所有窗口

