

"""车辆类，维护速度信息，并做出控制"""
import numpy as np


class Vehicle(object):
    def __init__(self):
        self.max_v = 1          # 最大速度
        self.max_ang_v = 0.25   # 最大角速度
        self.angle_threshold = 5 * np.pi * self.max_ang_v / 19   # 角度调整阈值

    def control_algo1(self, x, z, rotation_matrix, target):
        """尾部向前的控制逻辑"""
        # 测试了一下，基本以0.2转一圈需要100帧，每次只能移动0.0628rad
        # 求出与目标差距角度angle
        if rotation_matrix[0][2] <= 0:
            angle_old = np.arccos(rotation_matrix[0][0])
        else:
            angle_old = 2 * np.pi - np.arccos(rotation_matrix[0][0])
        # 上部分代码的逻辑是由于返回弧度差的范围不一样，所以需要统一范围，并且当代码在3.14附近时会出现突变，需要特殊讨论
        angle_need = np.arctan2(target[1] - z, target[0] - x)
        if angle_need < 0:
            angle_need = 2 * np.pi + angle_need
        angle = (angle_need - angle_old)

        if angle > self.angle_threshold:
            motor_velocity = [self.max_ang_v, -self.max_ang_v, self.max_ang_v, -self.max_ang_v]
            action_type = [2, self.max_ang_v]
        elif angle < -self.angle_threshold:
            motor_velocity = [-self.max_ang_v, self.max_ang_v, -self.max_ang_v, self.max_ang_v]
            action_type = [2, -self.max_ang_v]  # 2是旋转
        else:
            motor_velocity = [-self.max_v, -self.max_v, -self.max_v, -self.max_v]
            action_type = [1, -self.max_v]  # 1是直行

        print("动作", action_type, end=' ')
        return motor_velocity, action_type
    #
    # def control_algo2(self, x, z, rotation_matrix, target):
    #     """头部向前的控制逻辑"""
    #     if rotation_matrix[0][2] <= 0:
    #         angle_old = np.arccos(rotation_matrix[0][0])
    #     else:
    #         angle_old = 2 * np.pi - np.arccos(rotation_matrix[0][0])
    #     # 上部分代码的逻辑是由于返回弧度差的范围不一样，所以需要统一范围，并且当代码在3.14附近时会出现突变，需要特殊讨论
    #     angle_need = np.arctan2(target[1] - z, target[0] - x)
    #     if angle_need < 0:
    #         angle_need = 2 * np.pi + angle_need
    #     angle = (angle_need - angle_old) - np.pi
    #
    #     if angle > self.angle_threshold:
    #         motor_velocity = [self.max_ang_v, -self.max_ang_v, self.max_ang_v, -self.max_ang_v]
    #         action_type = [2, self.max_ang_v]
    #     elif angle < -self.angle_threshold:
    #         motor_velocity = [-self.max_ang_v, self.max_ang_v, -self.max_ang_v, self.max_ang_v]
    #         action_type = [2, -self.max_ang_v]
    #     else:
    #         motor_velocity = [self.max_v, self.max_v, self.max_v, self.max_v]
    #         action_type = [1, self.max_v]
    #
    #     print("动作", action_type, end='  ')
    #     return motor_velocity, action_type

    def vehicle_modify_pts(self, pts_count):
        """根据提取到的特征点数目，决定车辆速度"""
        if pts_count >= 120:
            self.max_v = 1
            self.max_ang_v = 0.25
        elif 70 < pts_count < 120:
            self.max_v = 0.5
            self.max_ang_v = 0.15
        else:
            self.max_v = 0.25
            self.max_ang_v = 0.1
        self.angle_threshold = 5 * np.pi * self.max_ang_v / 19

    def vehicle_modify_ratio(self, ratio):
        """如果发现光流追踪到的比例小于某一阈值，马上降速"""
        if ratio < 55:
            self.max_v = min(0.5, self.max_v)
            self.max_ang_v = min(0.15, self.max_v)
            self.angle_threshold = 5 * np.pi * self.max_ang_v / 19

