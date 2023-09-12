
"""视觉里程计，完成特征提取、追踪与定位"""

import numpy as np
import cv2
from algorithm.utils import pixel2cam


class Odometer(object):
    def __init__(self):
        # 提取出的3d-2d点对，特征点数量
        self.kp = []
        self.kp_3d = []
        self.pts_count = 0
        # 实际追踪到的将用于pnp的3d-2d点对
        self.pts_3d = []
        self.pts_2d = []
        # 相机选择，0是前相机，1是后相机
        self.cam_choose = 1
        # 前一帧图片
        self.img_f_1 = None  # 前一帧前相机图像（灰度图）
        self.depth_f_1 = None  # 前一帧前相机深度（灰度图）
        self.img_b_1 = None
        self.depth_b_1 = None

        # params for ShiTomasi corner detection
        self.feature_params = dict(maxCorners=1000, qualityLevel=0.01, minDistance=5, blockSize=7)
        # Parameters for lucas kanade optical flow
        self.lk_params = dict(winSize=(21, 21), maxLevel=4,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        # 相机内参和畸变
        self.K = np.array([[640.5098521801531, 0, 640.0],
                           [0, 640.5098521801531, 360.0],
                           [0, 0, 1]])
        self.distCoeffs = np.zeros(4)  # 无畸变[0. 0. 0. 0.]

    def feature_extract(self, img_f, img_b, depth_f, depth_b):
        """输入灰度图片，提取特征，保存kp_2d和相机系的kp_3d，并返回可供追踪的特征点数目和选择的相机"""
        self.kp = []
        self.kp_3d = []
        self.pts_count = 0
        # 提取出要去追踪的特征点及对应的相机系坐标
        pts_b = cv2.goodFeaturesToTrack(img_b, mask=None, **self.feature_params)
        pts_f = cv2.goodFeaturesToTrack(img_f, mask=None, **self.feature_params)
        if pts_b is None or pts_f is None:
            return 0, 0

        pts_b = np.float32(pts_b).reshape(-1, 2)
        pts_f = np.float32(pts_f).reshape(-1, 2)

        kp_b = []
        kp_b_3d = []
        kp_f = []
        kp_f_3d = []  # u,v 和 u,v,d
        if (len(pts_b) < 16) and (len(pts_f) < 16):   # 如果点数太少，一般也提不到多少深度正常的点。直接去预测
            # print("报错位点7，feature_extract特征点", len(pts_b))
            return 0, 0
        else:
            for (u, v) in pts_b:        # 查看后相机的有效特征点数
                d = depth_b[int(u)][int(v)]
                if d < 10:
                    kp_b.append([u, v])
                    kp_b_3d.append([u, v, d + 0.4])
            for (u, v) in pts_f:         # 查看前相机的有效特征点数
                d = depth_f[int(u)][int(v)]
                if d < 10:
                    kp_f.append([u, v])
                    kp_f_3d.append([u, v, d + 0.4])

        b_len = len(kp_b)
        f_len = len(kp_f)
        if (b_len < 8) and (f_len < 8):     # 有深度的点太少则直接去预测
            # print("报错位点8，深度才提取到不到8对点")
            return 0, 0
        else:
            if b_len >= f_len:
                self.kp = kp_b
                self.pts_count = b_len
                self.cam_choose = 1
                for (u, v, dd) in kp_b_3d:
                    self.kp_3d.append(pixel2cam(u, v, dd))
            else:
                self.kp = kp_f
                self.pts_count = f_len
                self.cam_choose = 0
                for (u, v, dd) in kp_f_3d:
                    self.kp_3d.append(pixel2cam(u, v, dd))

        # print("len", self.pts_count, end=" ")
        return self.pts_count, self.cam_choose

    def optical_flow(self, img_f_2, img_b_2, flag):
        """传入待追踪的前后相机灰度图，函数自动选择对应的相机进行追踪。
        flag=1代表追踪前一帧的特征点，flag=0代表追踪后一帧的特征点
        返回：追踪到的点数的百分比"""
        if not self.kp:
            # print("报错位点6，这是特征点不足引起的")
            return False, 0
        self.pts_3d = []
        self.pts_2d = []
        kp = np.array(self.kp)
        if flag:
            if self.cam_choose:  # 使用后相机
                pt, status, _err = cv2.calcOpticalFlowPyrLK(img_b_2, self.img_b_1, kp, None, **self.lk_params)
                if len(pt) < 8:
                    # print("报错位点10，有特征点，但是在下一帧图基本没追到")
                    return False, 0
                pt_r, status_r, _err = cv2.calcOpticalFlowPyrLK(self.img_b_1, img_b_2, pt, None, **self.lk_params)
            else:
                pt, status, _err = cv2.calcOpticalFlowPyrLK(img_f_2, self.img_f_1, kp, None, **self.lk_params)
                if len(pt) < 8:
                    # print("报错位点10，有特征点，但是在下一帧图基本没追到")
                    return False, 0
                pt_r, status_r, _err = cv2.calcOpticalFlowPyrLK(self.img_f_1, img_f_2, pt, None, **self.lk_params)
        else:
            if self.cam_choose:  # 使用后相机
                pt, status, _err = cv2.calcOpticalFlowPyrLK(self.img_b_1, img_b_2, kp, None, **self.lk_params)
                if len(pt) < 8:
                    # print("报错位点10，有特征点，但是在下一帧图基本没追到")
                    return False, 0
                # back-tracking for match verification
                pt_r, status_r, _err = cv2.calcOpticalFlowPyrLK(img_b_2, self.img_b_1, pt, None, **self.lk_params)
            else:
                pt, status, _err = cv2.calcOpticalFlowPyrLK(self.img_f_1, img_f_2, kp, None, **self.lk_params)
                if len(pt) < 8:
                    # print("报错位点10，有特征点，但是在下一帧图基本没追到")
                    return False, 0
                pt_r, status_r, _err = cv2.calcOpticalFlowPyrLK(img_f_2, self.img_f_1, pt, None, **self.lk_params)

        for i in range(self.pts_count):
            if status[i] and status_r[i]:
                pt_error = kp[i] - pt_r[i]
                if (pt_error[0] ** 2 + pt_error[1] ** 2) < 0.5:
                    self.pts_3d.append(self.kp_3d[i])
                    self.pts_2d.append(pt[i])

        print("点对：", len(self.pts_2d), end="  ")
        # 追踪到的点数的百分比
        track_ratio = round(100 * len(self.pts_2d) / self.pts_count, 2)
        print("追踪率", track_ratio, end=" ")
        return True, track_ratio

    def estimate_pose(self, flag):
        """使用pnp对相机运动进行估计, flag=1表示估计的是Tcc1，否则为Tc1c
        返回：前（后）相机系的运动Tcc1"""
        T = np.eye(4)
        if len(self.pts_2d) < 8:  # 不进行估计，直接预测
            # print("报错位点4，用于EPNP的点数太少")
            return False, T
        (success, rvec, tvec, _) = cv2.solvePnPRansac(np.array(self.pts_3d), np.array(self.pts_2d), self.K,
                                                      self.distCoeffs, flags=cv2.SOLVEPNP_EPNP)  # 0.0005617
        # (success, rvec, tvec) = cv2.solvePnP(np.array(self.pts_3d), np.array(self.pts_2d), self.K,
        #                                               self.distCoeffs, flags=cv2.SOLVEPNP_EPNP)
        if success:
            rotM = cv2.Rodrigues(rvec)[0]
            T[:3, :3] = rotM
            T[:3, 3] = np.array(tvec).reshape(3)
        if flag:
            return success, T
        else:
            return success, np.linalg.inv(T)
