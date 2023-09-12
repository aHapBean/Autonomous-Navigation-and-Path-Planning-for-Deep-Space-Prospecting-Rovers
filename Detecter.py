
"""检测器，用于检测障碍物和目标箱体"""

import numpy as np
import cv2


class Detecter(object):
    def __init__(self):
        self.low_hsv = np.uint8([0, 80, 80])
        self.high_hsv = np.uint8([25, 254, 255])
        self.kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))    # 开运算先腐蚀后膨胀，核要小。因为噪声本身不大
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))  # 闭运算先膨胀后腐蚀，核要大一点，不然连不起来

        self.neighborhood_size = 21  # 奇数！！用来识别障碍物轮廓是否有效的的邻域大小
        self.window_width = 600  # 偶数！！识别障碍只看正前方，故选取相机视野靠近中心部分的一个窗口，宽400高100
        self.window_height = 300  # 偶数！！
        self.bigwindow_width = self.window_width + self.neighborhood_size - 1
        self.bigwindow_height = self.window_height + self.neighborhood_size - 1
        self.row_center = 500  # 以(center行，640列)为中心，扩展出一个window来
        self.chubutance = 600
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        self.depth_canny_low = 180
        self.depth_ratio = 2
        self.threshold_low_detect = 0.9
        self.threshold_high_detect = 1.2
        self.threshold_low_info = 0.8
        self.threshold_high_info = 2

    def detect_targets(self, rgb_f, rgb_b):
        """传入前后相机的灰度图，返回是否检测到箱体，及对应的像素坐标"""
        target_existence = 0  # “0”代表未发现目标，"1"代表前相机发现目标，“2”代表后相机发现目标，“3”代表前后都发现目标

        """RGB转HSV，用颜色识别target"""
        hsv_f = cv2.cvtColor(rgb_f, cv2.COLOR_RGB2HSV)
        hsv_b = cv2.cvtColor(rgb_b, cv2.COLOR_RGB2HSV)

        mask_f = cv2.inRange(hsv_f, lowerb=self.low_hsv, upperb=self.high_hsv)
        mask_b = cv2.inRange(hsv_b, lowerb=self.low_hsv, upperb=self.high_hsv)
        centroids_f = []
        centroids_b = []

        if mask_b.sum() > 0:  # 如果没检测到目标，跳过后续处理，加快速度
            target_existence = 2
            # 先开运算后闭运算，即先去除噪声，再把目标内部的空缺补上
            mask_b = cv2.morphologyEx(cv2.morphologyEx(mask_b, cv2.MORPH_OPEN, self.kernel_open), cv2.MORPH_CLOSE,
                                      self.kernel_close)
            # 连通域数量、连通域矩阵、外接矩形、连通域质心
            retval_b, labels_b, stats_b, centroids_b = cv2.connectedComponentsWithStats(mask_b, connectivity=8)
            # retval_b -= 1   # 去掉“背景”这个连通域
            centroids_b = np.delete(np.uint16(centroids_b), 0, 0)

        if mask_f.sum() > 0:
            if target_existence == 2:
                target_existence = 3
            else:
                target_existence = 1
            mask_f = cv2.morphologyEx(cv2.morphologyEx(mask_f, cv2.MORPH_OPEN, self.kernel_open), cv2.MORPH_CLOSE,
                                      self.kernel_close)  # 先开运算后闭运算，即先去除噪声，再把目标内部的空缺补上
            # 连通域数量、连通域矩阵、外接矩形、连通域质心
            retval_f, labels_f, stats_f, centroids_f = cv2.connectedComponentsWithStats(mask_f, connectivity=8)
            centroids_f = np.delete(np.uint16(centroids_f), 0, 0)

        """显示相机探测到的目标黑白图像"""
        # cv2.imshow("target_bw_f", mask_f)
        # cv2.imshow("target_bw_b", mask_b)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print(target_existence, centroids_f, centroids_b)
        """centroids_f, centroids_b分别是前后RGB相机探测到的各个目标箱子在画面中形心的像素坐标，所形成的二维nparray"""
        return target_existence, centroids_f, centroids_b

    def detect_obstacles(self, d_image_b):
        """该函数用来检测障碍物，并返回障碍物的box及其平均深度"""
        obstacle_existence = False
        dep_b = np.array(d_image_b)
        dep_b = np.nan_to_num(dep_b, nan=0, posinf=0.4, neginf=0.4) + 0.4
        dep_b = np.transpose(dep_b)
        dep_chubu = dep_b[self.chubutance, 640 - self.bigwindow_width // 2:640 + self.bigwindow_width // 2]
        dep_window0 = dep_b[self.row_center - self.bigwindow_height // 2:self.row_center + self.bigwindow_height // 2,
                      640 - self.bigwindow_width // 2:640 + self.bigwindow_width // 2]
        dep_window = np.empty((self.bigwindow_height, self.bigwindow_width))

        if np.min((dep_chubu - self.threshold_low_detect) * (
                dep_chubu - self.threshold_high_detect)) > 0:  # 视野内的东西都在1.3m开外，无需避障
            box = []
            depth = -1
            return obstacle_existence, box, depth
        else:  # 需避障
            for i in range(self.bigwindow_height):
                for j in range(self.bigwindow_width):
                    if dep_window0[i][j] <= self.threshold_low_info:
                        dep_window[i][j] = 0
                    elif dep_window0[i][j] >= self.threshold_high_info:
                        dep_window[i][j] = 255
                    else:
                        dep_window[i][j] = 250 / (self.threshold_high_info - self.threshold_low_info) * (
                                    dep_window0[i][j] - self.threshold_high_info)  # 线性变换，便于边缘检测
            dep_window = dep_window.astype(np.uint8)

            # 使用Canny边缘检测算法
            dep_edges = cv2.Canny(dep_window, self.depth_canny_low, self.depth_ratio * self.depth_canny_low)
            dep_closed = cv2.morphologyEx(dep_edges, cv2.MORPH_CLOSE, self.kernel)

            # 寻找轮廓
            dep_contours, _ = cv2.findContours(dep_closed.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            dep_selected = np.zeros((self.bigwindow_height, self.bigwindow_width), dtype=np.uint8)
            # 绘制轮廓
            for contour in dep_contours:
                # 计算轮廓的长度
                arc_length = cv2.arcLength(contour, False)
                # 根据长度设定阈值，过滤掉过小的轮廓
                if arc_length < 100:
                    # 绘制轮廓
                    continue
                else:
                    hull = cv2.convexHull(contour)
                    area = cv2.contourArea(hull)
                    perimeter = cv2.arcLength(hull, True)  # 第二个参数表示凸包是否闭合，如果是闭合的凸包则为 True，否则为 False
                    if perimeter > 0 and area / perimeter > 10:
                        cv2.drawContours(dep_selected, [contour], -1, 255, thickness=2)
            # 消除假边缘
            obstacle = dep_selected.copy()
            if obstacle.sum() > 0:
                non_zero_pixels = cv2.findNonZero(obstacle)
                for pixel in non_zero_pixels:
                    x, y = pixel[0]
                    # 获取当前像素的邻域范围
                    y_start = max(0, y - self.neighborhood_size // 2)
                    y_end = min(dep_window.shape[0], y + self.neighborhood_size // 2 + 1)
                    x_start = max(0, x - self.neighborhood_size // 2)
                    x_end = min(dep_window.shape[1], x + self.neighborhood_size // 2 + 1)
                    if x_start >= x_end or y_start >= y_end or y_end > dep_window.shape[0] or x_end > dep_window.shape[
                        1]:
                        continue
                    # 获取邻域内其他像素的信息
                    neighborhood = dep_window.copy()[y_start:y_end, x_start:x_end]
                    if neighborhood.min() == 0:
                        if neighborhood.max() > 245:
                            obstacle[y, x] = 0

            if obstacle.sum() > 0:
                non_zero_pixels = cv2.findNonZero(obstacle)
                rect = cv2.minAreaRect(non_zero_pixels)
                if np.min(rect[1]) < 35:
                    box = []
                    depth = -1
                    return obstacle_existence, box, depth
                else:
                    obstacle_existence = True
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    centroid = np.int0(np.mean(box, axis=0))
                    depth = dep_window0[int(centroid[1]), int(centroid[0])]
                    if depth > self.threshold_high_info:
                        depth = self.threshold_low_detect
                    cv2.drawContours(obstacle, [box], 0, 255, 2)
                    box[:, 0] += 640 - self.bigwindow_width // 2
                    box[:, 1] += self.row_center - self.bigwindow_height // 2
                    # plt.subplot(2,2,1)
                    # plt.imshow(dep_window0)
                    # plt.subplot(2, 2, 2)
                    # plt.imshow(dep_edges)
                    # plt.subplot(2,2,3)
                    # plt.imshow(obstacle)
                    # plt.subplot(2,2,4)
                    # plt.imshow(dep_window)
                    # plt.show()
            else:
                box = []
                depth = -1
                return obstacle_existence, box, depth

        # """box是检测到障碍物边缘的最小外接四边形的四个顶点的像素坐标构成的二维nparray,顺序是左上角-右上角-右下角-左下角，[像素点所在列数,像素点所在行数]。如果未检测到障碍物边缘，则box的值为[[-1,-1]]"""
        # """取四个顶点坐标可用以下代码"""
        # # if box.shape[0] == 4:
        # #     left_up_hang = box[0,1]
        # #     left_up_lie = box[0,0]
        # #     right_up_hang = box[1,1]
        # #     right_up_lie = box[1,0]
        # #     right_down_hang = box[2,1]
        # #     right_down_lie = box[2,0]
        # #     left_down_hang = box[3,1]
        # #     left_down_lie = box[3,0]
        # """"""
        # print(box)
        return obstacle_existence, box, depth

    def detect_depth(self, d_img_f, d_img_b, criteria):
        """该函数用来检测深度图信息是否丰富，如果信息非常丰富，可以用来局部与全局点云配准"""
        d_f = np.asarray(np.array(d_img_f) < 10).nonzero()
        d_b = np.asarray(np.array(d_img_b) < 10).nonzero()
        if d_b[1].size > criteria[0]:   # 先看后相机视野够不够丰富
            v_range_b = np.max(d_b[1]) - np.min(d_b[1])
            if v_range_b > criteria[1]:
                print("配准开始", d_b[1].size, v_range_b)
                return True, d_f, d_b
        elif d_f[1].size > criteria[0]:
            v_range_f = np.max(d_f[1]) - np.min(d_f[1])
            if v_range_f > criteria[1]:
                print("配准开始", d_f[1].size, v_range_f)
                return True, d_f, d_b

        return False, [], []


