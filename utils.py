

"""部分通用函数"""

import numpy as np
import open3d as o3d
import math
from scipy.interpolate import griddata

# 相机参数
cx = 640.0
cy = 360.0
fx = 640.5098521801531
fy = 640.5098521801531


def pixel2cam(u, v, d):
    x_c = d * (u - cx) / fx
    y_c = d * (v - cy) / fy
    return [x_c, y_c, d]


def t_q_to_T(q, translation):
    """由四元数和平移向量获得欧式矩"""
    w, x, y, z = q
    rotation_matrix = np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]
    ])
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = translation
    return T


def bias_calculate(new_tvec, old_tvec):
    error = new_tvec - old_tvec  # 与前一时刻的位移
    return math.sqrt(error[0] ** 2 + error[1] ** 2 + error[2] ** 2)


def judge_pose_whether_right(new_tvec, old_tvec, action_type):
    """传入估计的最新xyz，判断其是否符合要求"""
    # 只有误差在允许范围内才可以更新self参量
    bias = bias_calculate(new_tvec, old_tvec)
    type_a = action_type[0]
    action = action_type[1]
    if type_a == 1:  # 尾向直行
        criteria = 3.5 * abs(action) / 10
        print("位姿", new_tvec, end="  ")
        if bias < criteria:
            return True
    else:  # 旋转
        print("位姿", new_tvec, end="  ")
        if bias < 0.15:
            return True

    return False


def generate_target_pc(moon_map):
    # 生成格式为xyz的点云
    points = []
    values = []
    for z in range(500):
        for x in range(500):
            points.append([float(x) / 10 - 25, float(z) / 10 - 25])
            values.append(moon_map[z][x][3])
    # 定义插值网格的网格点
    x = np.linspace(-25.0, 25.0, num=1000)  # 在X轴方向上创建插值网格点
    z = np.linspace(-25.0, 25.0, num=1000)  # 在Z轴方向上创建插值网格点
    xx, zz = np.meshgrid(x, z, indexing='ij')  # 创建网格坐标矩阵
    # 执行线性插值
    interpolated_values = griddata(points, np.array(values), (xx, zz), method='linear')
    interpolated_data = np.column_stack((np.ravel(xx), np.ravel(zz), np.ravel(interpolated_values)))
    interpolated_data[:, [1, 2]] = interpolated_data[:, [2, 1]]
    target_cloud = o3d.geometry.PointCloud()
    target_cloud.points = o3d.utility.Vector3dVector(interpolated_data)
    # o3d.io.write_point_cloud("target_inter.ply", point_cloud)
    # o3d.visualization.draw_geometries([point_cloud])

    return target_cloud
