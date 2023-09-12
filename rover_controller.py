"""
@author: 15201622364
"""
import cv2
import joblib
import numpy as np
import open3d as o3d
import pandas as pd
import algorithm.utils as utils
from matplotlib import pyplot as plt
from algorithm.global_planner import global_planner
from algorithm.Odometer import Odometer
from algorithm.Vehicle import Vehicle
from algorithm.Detecter import Detecter
from algorithm.yolov5.detectinterest import detect


class rover_controller(object):
    def __init__(self, moon_map, pos_A, area_B):
        """全局与初始信息"""
        self.map = moon_map
        self.pos_A = pos_A
        self.area_B = area_B
        x_map = (25 + pos_A[0]) * 10  # 小车初始在moon_map上的x坐标（横坐标）
        z_map = (25 - pos_A[1]) * 10  # 小车初始在moon_map上的z坐标（纵坐标）

        """环境状态参量"""
        self.runtime = 0    # 全局时间
        self.done = False
        self.env_state = False  # 为确保程序正确运行，跳过第一帧可能失误的画面
        self.arrive_time = 0

        """初始rover位姿和相机参量"""
        rover_t = [pos_A[0], moon_map[z_map][x_map][3], -pos_A[1]]  # TODO:轮子高度还不知,wbt地图也不准确
        # rover_t = [10.0223, 0.738952, 14.9047]
        rover_q = [0.705929, -0.0294136, -0.707469, 0.0169222]  # TODO:如何初始化它？
        self.Twr = utils.t_q_to_T(rover_q, rover_t)  # 小车初始欧式矩阵
        self.rover_t = np.array(rover_t)
        # rgb相机相对于rover的相关参数
        camf_t = [-0.1895, 0, 0]
        camf_q = [0.696366, -0.122787, 0.696363, 0.122787]  # camf相对rover的四元数，定值
        T_cam2std = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        _Trc_f = utils.t_q_to_T(camf_q, camf_t)  # webots中相机相对rover的位姿，但不是标准相机系
        self.Trcf = np.dot(_Trc_f, T_cam2std)  # 标准rgb相机系
        self.Tcfr = np.linalg.inv(self.Trcf)
        camb_t = [0.1895, 0, 0]
        camb_q = [0.696362, 0.122787, -0.696366, 0.122788]  # camb相对rover的四元数，定值
        _Trc_b = utils.t_q_to_T(camb_q, camb_t)
        self.Trcb = np.dot(_Trc_b, T_cam2std)
        self.Tcbr = np.linalg.inv(self.Trcb)

        """路径规划器"""
        start_point = [x_map, z_map]  # 500 x 500
        # self.end_point = [(25 + area_B[0]) * 10, (25 - area_B[1]) * 10]
        area_B_for_planner = [(25 + area_B[0]) * 10, (25 - area_B[1]) * 10, area_B[2] * 10, area_B[3] * 10]
        self.global_planner = global_planner(moon_map, start_point, area_B_for_planner)
        """终点策略"""
        self.area_point_list = self.global_planner.get_final_path_list(middle_flag=False)
        print("终点阶段点：", self.area_point_list)
        self.area_point_ind = 0
        self.end_point = self.area_point_list[self.area_point_ind]
        # 全局路径初始化时生成，global_path[0]是起点
        print("———————global path planning————————")
        temp_global_path = self.global_planner.with_astar_deemed_to_plan_path(start_point, self.end_point)
        self.global_path = np.array(temp_global_path)
        self.path_idx = 1       # waypoint位点

        """巡视器状态和指标"""
        self.rover_state = 0    # 1是纯旋转状态，用于在关键点旋转一周。2是避障状态
        self.rotate_count = 80  # 纯旋转的次数，大概原地转75次

        """车辆控制器"""
        self.vehicle = Vehicle()
        self.motor_velocity = [0, 0, 0, 0]
        self.action_type = None  # 动作类型

        """视觉里程计控制器"""
        self.odometer = Odometer()
        self.cam_choose = 0  # 相机使用，0为前相机，1为后相机
        # self.random_forest = joblib.load('algorithm/random_forest_model.pkl')
        self.need_predict = False

        """点云配准相关参数"""
        self.target_pc = utils.generate_target_pc(moon_map)   # 全局地图的点云
        # o3d.io.write_point_cloud("target.ply", self.target_pc)
        self.need_register = False    # 是否需要点云配准
        self.max_peace_time = 60
        self.max_wait_register = 70   # 当需要进行点云配准时，开始寻找视野比较广的帧，如果100帧还找不到，则降低标准
        self.register_criteria = [170000, 300]   # 即至少200000个点，300的v宽度，才算信息丰富的深度图
        self.register_interval = 20     # 两次配准间隔至少12帧
        self.register_time = 0          # 记录配准时的全局时间
        self.tmp_instruct = True

        """目标探测器及检测间隔参数"""
        self.detecter = Detecter()
        self.color_detect_interval = 3
        self.yolo_detect_interval = 6
        self.target_pos = []         # 目标箱体位置
        # 避障状态检测参数
        self.obstacle_detect_interval = 4
        self.avoid_wait_interval = 25  # 如果处于避障状态，则25帧内不再检测障碍物

        self.trajectory_for_reality = []   # 存储位置，为了demo
        self.trajectory_for_virtual = []

    def step(self, rgb_image_f, rgb_image_b, d_image_f, d_image_b, world_time, position):
        """为确保图像正确传输的原地等待操作"""
        if rgb_image_f is None:
            print("环境返回空图像，保持不动，等待下一帧")
            return [0, 0, 0, 0], False, []
        elif self.env_state is not True:     # 为防止环境运用方法不同，第一次传回图片也跳过不处理，等待环境完全准备好
            self.env_state = True
            return [0, 0, 0, 0], False, []

        self.runtime += 1
        print("world_time:", world_time, end=' ')

        if (self.runtime - self.arrive_time) > 2000:        # 如果长时间没到下一个目标点，退出
            print("TMD跑了一年还没到最后一个点，中断")
            print('self.area_point_ind', self.area_point_ind)
            return [0, 0, 0, 0], True, self.target_pos
        """处理图片格式"""
        rgb_f = np.uint8(rgb_image_f).transpose(1, 0, 2)        # (720, 1280, 3)彩色图，层顺序是RGB
        rgb_b = np.uint8(rgb_image_b).transpose(1, 0, 2)
        rgb_f_gray = cv2.cvtColor(rgb_f, cv2.COLOR_RGB2GRAY)  # (720, 1280)的灰度图
        rgb_b_gray = cv2.cvtColor(rgb_b, cv2.COLOR_RGB2GRAY)

        save_path = rf'C:\tiaozhanbei\simenv\rgb_b\{world_time}.png'
        plt.imsave(save_path, rgb_b)
        """位姿更新——特征提取+光流追踪+pnp"""
        if self.runtime > 1:
            flag = 0   # 标识追踪状态，0代表追踪后一帧，1代表追踪前一帧
            if self.runtime % 2 == 0:
                flag = 1       # 偶数倍提取特征，并追踪前一帧的位姿
                pts_count, self.cam_choose = self.odometer.feature_extract(rgb_f_gray, rgb_b_gray, d_image_f, d_image_b)
                if pts_count == 0:     # 提取特征失败，进行预测
                    print("位点1， 提取特征失败")
                    self.predict_pose_by_motion()
                else:
                    self.vehicle.vehicle_modify_pts(pts_count)
                    if (pts_count > 650) and ((self.runtime - self.register_time) > (self.register_interval - 5)):
                        # 特征点相当丰富，可能有山，马上抬高标准，尝试搜索最佳帧
                        self.need_register = True
                        self.register_criteria = [135000, 320]

            suc_flow, track_ratio = self.odometer.optical_flow(rgb_f_gray, rgb_b_gray, flag)
            if suc_flow:
                self.vehicle.vehicle_modify_ratio(track_ratio)      # 根据追踪率调整速度
                suc_pose, Tcc1 = self.odometer.estimate_pose(flag)
                if suc_pose:    # 如果里程计成功预测了位姿变化，就判断其正确与否
                    if self.cam_choose:  # 后相机
                        Twr_new = self.Twr @ self.Trcb @ Tcc1 @ self.Tcbr
                    else:
                        Twr_new = self.Twr @ self.Trcf @ Tcc1 @ self.Tcfr

                    new_tvec = Twr_new[:3, 3]
                    if utils.judge_pose_whether_right(new_tvec, self.rover_t, self.action_type):  # 判断是否在合理范围内
                        self.Twr = Twr_new
                        self.rover_t = new_tvec
                    else:
                        self.predict_pose_by_motion()
                        print("位点20，认为pnp估计的不对")
                else:   # 否则直接用运动预测
                    self.predict_pose_by_motion()
                    print("位点2，pnp返回失败")
            else:
                self.predict_pose_by_motion()
                print("位点3，光流返回失败")

        # 更新odometer的灰度图片
        self.odometer.img_f_1 = rgb_f_gray
        self.odometer.img_b_1 = rgb_b_gray
        self.odometer.depth_f_1 = d_image_f
        self.odometer.depth_b_1 = d_image_b

        """如果深度图像丰富，做一次局部与全局点云配准"""
        if self.need_register:
            self.max_wait_register -= 1
            if self.max_wait_register == 30:    # 如果时间到了还找不到最佳观测帧，就降低标准
                self.register_criteria = [135000, 230]
            elif self.max_wait_register == 0:
                self.register_criteria = [100000, 200]
                print("点云配准标准降低")
            if self.runtime % 4 == 0:  # 每隔三帧探测一下深度信息如何
                suc_dep, d_f_index, d_b_index = self.detecter.detect_depth(d_image_f, d_image_b, self.register_criteria)
                if suc_dep:          # 找到了合适的深度图
                    self.local_global_register(d_f_index, d_b_index, d_image_f, d_image_b)
                    # 重新规划到终点的路径
                    self.global_path = np.array(self.global_planner.with_astar_deemed_to_plan_path(
                        [self.rover_t[0]*10 + 250, self.rover_t[2] * 10 + 250], self.end_point))
                    self.path_idx = 1
                    print("重新规划的路径", self.global_path)
        else:
            if (self.runtime - self.register_time) > self.max_peace_time:  # 如果一直运行平稳，则最多180帧后去寻找配准点
                self.need_register = True
                print("平稳运行太久了，准备找机会来个点云配准吧")

        # 目前机器人的位置
        x = (self.rover_t[0] + 25) * 10
        z = (self.rover_t[2] + 25) * 10
        self.trajectory_for_virtual.append(position)
        self.trajectory_for_reality.append([x, z])

        # 障碍物检测与局部路径规划，如果处于避障状态，等待避障周期结束
        if self.rover_state == 2:
            if self.avoid_wait_interval:
                self.avoid_wait_interval -= 1
            else:
                self.avoid_wait_interval = 25
                self.rover_state = 0            # 25帧过后允许再次识别障碍物
                print("避障系统恢复，可再次避障")

        elif not self.rover_state:     # 如果不处于避障和纯旋转状态，则每隔2帧检测一次障碍物
            if self.runtime % self.obstacle_detect_interval == 0:
                obstacle_existence, obstacle_box, obstacle_depth = self.detecter.detect_obstacles(d_image_b)
                if obstacle_existence:  # 检测到障碍物了
                    self.rover_state = 2        # 避障状态
                    x_obstacle = []
                    z_obstacle = []
                    # 寻找障碍物尺寸，并适当扩大防止碰撞
                    for (u, v) in obstacle_box:
                        p_front = self.pixel2world(u, v, obstacle_depth, 1)
                        p_back = self.pixel2world(u, v, obstacle_depth + 0.8, 1)
                        x_obstacle.append(p_front[0])
                        x_obstacle.append(p_back[0])
                        z_obstacle.append(p_front[2])
                        z_obstacle.append(p_back[2])
                    # x_min = np.min(np.array(x_obstacle)) * 10 + 250 - 1
                    # x_max = np.max(np.array(x_obstacle)) * 10 + 250 + 1
                    # z_min = np.min(np.array(z_obstacle)) * 10 + 250 - 1
                    # z_max = np.max(np.array(z_obstacle)) * 10 + 250 + 1
                    x_min = np.min(np.array(x_obstacle)) * 10 + 250 + 1
                    x_max = np.max(np.array(x_obstacle)) * 10 + 250 - 1
                    z_min = np.min(np.array(z_obstacle)) * 10 + 250 + 1
                    z_max = np.max(np.array(z_obstacle)) * 10 + 250 - 1
                    box = [x_min, x_max, z_min, z_max]

                    # 局部避障
                    print("DEBUG: obstacle avoidance! local path generating…… ")
                    self.global_planner.local_planner_infomation(box)
                    self.global_path = np.array(
                        self.global_planner.with_astar_deemed_to_plan_path([x, z], self.end_point))
                    print("避障时生成的路径", self.global_path)
                    self.path_idx = 1
                    # print(self.global_path)

        """物体识别，颜色识别+yolo"""
        # if self.runtime % self.yolo_detect_interval == 0:
        #     # bgr_f = cv2.cvtColor(rgb_f, cv2.COLOR_RGB2BGR)
        #     bgr_b = cv2.cvtColor(rgb_f, cv2.COLOR_RGB2BGR)
        #     # detected_f = detect(bgr_f)
        #     detected_b = detect(bgr_b)
        #     # if detected_f:
        #     #     for (u, v) in detected_f:
        #     #         d = d_image_f[int(u)][int(v)]   # 因为这里d_image_f是竖着的，我没转置处理，所以对应的先索引u再索引v
        #     #         if d < 10:   # 深度正常才执行
        #     #             target_world = self.pixel2world(u, v, d + 0.4, 0)   # [x,y,z]
        #     #             self.add_to_target_pos(target_world)
        #     if detected_b:
        #         for (u, v) in detected_b:
        #             d = d_image_b[int(u)][int(v)]   # 因为这里d_image_f是竖着的，我没转置处理，所以对应的先索引u再索引v
        #             if d < 10:   # 深度正常才执行
        #                 target_world = self.pixel2world(u, v, d + 0.4, 1)   # [x,y,z]
        #                 self.add_to_target_pos(target_world)

        if self.runtime % self.color_detect_interval == 0:
            target_existence, centroids_f, centroids_b = self.detecter.detect_targets(rgb_f, rgb_b)
            if target_existence == 1:  # 前相机发现了
                for (u, v) in centroids_f:
                    dd = d_image_f[u][v] + 0.58  # 加0.58是考虑了箱子的厚度
                    if dd < 10:
                        target_world = self.pixel2world(u, v, dd, 0)
                        self.add_to_target_pos(target_world, world_time)
            elif target_existence == 2:  # 后相机发现了
                for (u, v) in centroids_b:
                    dd = d_image_b[u][v] + 0.58
                    if dd < 10:
                        target_world = self.pixel2world(u, v, dd, 1)
                        self.add_to_target_pos(target_world, world_time)
            elif target_existence == 3:  # 前后相机都发现了
                for (u, v) in centroids_f:
                    dd = d_image_f[u][v] + 0.58
                    if dd < 10:
                        target_world = self.pixel2world(u, v, dd, 0)
                        self.add_to_target_pos(target_world, world_time)
                for (u, v) in centroids_b:
                    dd = d_image_b[u][v] + 0.58
                    if dd < 10:
                        target_world = self.pixel2world(u, v, dd, 1)
                        self.add_to_target_pos(target_world, world_time)

        """控制与局部路径生成、系统运行结束标志"""
        # if self.rover_state == 1:        # 进入终点纯旋转探测目标阶段
        #     max_ang_v = self.vehicle.max_ang_v
        #     self.register_interval = 25
        #     if self.rotate_count:
        #         self.rotate_count -= 1
        #         self.motor_velocity = [-max_ang_v, max_ang_v, -max_ang_v, max_ang_v]
        #         self.action_type = [2, -max_ang_v]
        #     else:
        #         self.motor_velocity = [-max_ang_v, max_ang_v, -max_ang_v, max_ang_v]
        #         self.rover_state = 0
        #         self.rotate_count = 80
        # else:
        #     self.register_interval = 20
        while self.path_idx == len(self.global_path):
            print("DEBUG:到达该段的终点，进入纯旋转检测，并规划下一段111")
            self.arrive_time = self.runtime
            self.area_point_ind += 1
            if self.area_point_ind == 4:
                self.area_point_ind += 1
            self.yolo_detect_interval = 4  # 减小探测间隙，加快探测速度
            self.color_detect_interval = 2
            if self.area_point_ind == len(self.area_point_list):  # 程序结束的标志
                print("完全走完全程，程序结束111")
                print("运行时间111", world_time)
                self.done = True
                return [0, 0, 0, 0], self.done, self.target_pos
            self.end_point = self.area_point_list[self.area_point_ind]
            self.global_path = self.global_planner.without_astar_deemed_to_plan_path([x, z], self.end_point)
            self.path_idx = 1

        target_nav = self.global_path[self.path_idx]
        # 如果距离够小，则换新的目标点，否则还是用老的目标点
        if abs(x - target_nav[0]) < 2 and abs(z - target_nav[1]) < 2:
            self.path_idx += 1
            # 终点点列逻辑
            while self.path_idx == len(self.global_path):  # 因为有时路径长度规划出来是1，所以要用while
                print("DEBUG:到达该段的终点，进入纯旋转检测，并规划下一段222")
                self.arrive_time = self.runtime
                self.area_point_ind += 1
                if self.area_point_ind == 4:
                    self.area_point_ind += 1
                self.yolo_detect_interval = 4   # 减小探测间隙，加快探测速度
                self.color_detect_interval = 2
                if self.area_point_ind == len(self.area_point_list):        # 程序结束的标志
                    print("完全走完全程，程序结束222")
                    print("运行时间222", world_time)
                    np.savetxt('trajectory_for_virtual.txt', self.trajectory_for_virtual)
                    np.savetxt('trajectory_for_reality.txt', self.trajectory_for_reality)
                    self.done = True
                    return [0, 0, 0, 0], self.done, self.target_pos
                self.end_point = self.area_point_list[self.area_point_ind]
                self.global_path = self.global_planner.without_astar_deemed_to_plan_path([x, z], self.end_point)
                self.path_idx = 1
                # self.rover_state = 1     # 进入纯旋转状态
                # self.need_register = True  # 终点开启点云匹配修正误差
            target_nav = self.global_path[self.path_idx]

        rotation_matrix = self.Twr[:3, :3]
        self.motor_velocity, self.action_type = self.vehicle.control_algo1(x, z, rotation_matrix, target_nav)

        print("target:", self.target_pos)
        return self.motor_velocity, self.done, self.target_pos

    def add_to_target_pos(self, target, world_time):
        target_x = target[0]        # 对应说明上的坐标系
        target_z = -target[2]
        flag = True
        for (x, z) in self.target_pos:
            if ((x-target_x) ** 2 + (z-target_z) ** 2) < 4:   # 如果有目标与之相距2米以内，则不添加进target_pos
                flag = False
        if flag:
            self.target_pos.append([target_x, target_z])
            print("找到了", len(self.target_pos), "个箱子, 用时", world_time)

    def pixel2world(self, u, v, d, cam_choose):
        """像素坐标转世界坐标。p是像素坐标(u,v),dd是深度。
        公式为 pw = Twr * Trc * pc, 返回形式为[x, y, z]"""
        x_c, y_c, dd = utils.pixel2cam(u, v, d)
        p_cam = [x_c, y_c, dd, 1]
        if cam_choose:
            p_temp = np.dot(self.Trcb, p_cam)
        else:
            p_temp = np.dot(self.Trcf, p_cam)
        p_world = np.dot(self.Twr, p_temp)
        return p_world[:3]

    def predict_pose_by_motion(self):
        if (self.runtime - self.register_time) > self.register_interval:
            self.need_register = True    # 有丢帧的情况的话，需要在以后找机会进行配准
        if self.need_predict:
            self.need_predict = False
            v1 = np.array(self.motor_velocity[0])
            v2 = np.array(self.motor_velocity[1])
            v3 = np.array(self.motor_velocity[2])
            v4 = np.array(self.motor_velocity[3])

            r1 = np.array(self.Twr[0][0])
            r2 = np.array(self.Twr[0][1])
            r3 = np.array(self.Twr[0][2])
            r4 = np.array(self.Twr[1][0])
            r5 = np.array(self.Twr[1][1])
            r6 = np.array(self.Twr[1][2])
            r7 = np.array(self.Twr[2][0])
            r8 = np.array(self.Twr[2][1])
            r9 = np.array(self.Twr[2][2])

            data = pd.DataFrame({
                'v1': v1, 'v2': v2, 'v3': v3, 'v4': v4, 'r1': r1,
                'r2': r2, 'r3': r3, 'r4': r4, 'r5': r5, 'r6': r6, 'r7': r7,
                'r8': r8, 'r9': r9
            })
            X_test = data[['v1', 'v2', 'v3', 'v4', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9']]
            y_pred = self.random_forest.predict(X_test)

            ret = []
            for i in y_pred:
                for j in i:
                    ret.append(j)
            return ret

    def local_global_register(self, d_f_index, d_b_index, d_image_f, d_image_b):
        # 参数回归初始值
        print("""局部与全局点云配准优化""")
        self.register_time = self.runtime
        self.need_register = False
        self.register_criteria = [170000, 300]
        self.max_wait_register = 60        # 降低了一丢丢标准
        self.max_peace_time = 60
        source = []
        d_f_u = list(d_f_index)[0]
        d_f_v = list(d_f_index)[1]
        d_b_u = list(d_b_index)[0]
        d_b_v = list(d_b_index)[1]
        for (uf, vf, ub, vb) in zip(d_f_u, d_f_v, d_b_u, d_b_v):
            source.append(self.pixel2world(uf, vf, d_image_f[uf][vf] + 0.4, 0))
            source.append(self.pixel2world(ub, vb, d_image_b[ub][vb] + 0.4, 1))
        source_pc = o3d.geometry.PointCloud()
        source_pc.points = o3d.utility.Vector3dVector(np.array(source))
        source_pc = o3d.geometry.PointCloud.voxel_down_sample(source_pc, 0.1)
        threshold = 3.0  # 移动范围的阀值
        trans_init = np.asarray([[1, 0, 0, 0],  # 4x4 identity matrix，这是一个转换矩阵，
                                 [0, 1, 0, 0],  # 象征着没有任何位移，没有任何旋转，我们输入
                                 [0, 0, 1, 0],  # 这个矩阵为初始变换
                                 [0, 0, 0, 1]])
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source_pc, self.target_pc, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
        modify_T = reg_p2p.transformation

        # print("点云长度", len(source))
        # o3d.io.write_point_cloud("source{}.ply".format(self.runtime), source_pc)
        # print("点云预测的修正", reg_p2p.transformation)
        # print("原位姿", self.Twr)
        self.Twr = modify_T @ self.Twr
        self.rover_t = self.Twr[:3, 3]
        # print("修正位姿", self.Twr)
