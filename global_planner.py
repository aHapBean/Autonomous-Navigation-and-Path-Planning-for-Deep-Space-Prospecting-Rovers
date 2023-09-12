import math
import time
import random
import numpy as np
from queue import PriorityQueue
import matplotlib.pyplot as plt
import scipy.interpolate as scipy_interpolate
from scipy.special import comb
from math import floor


class global_planner:
    def __init__(self, moon_map, start_point, area_B):
        """
        :param moon_map:    未放大的初始地图
        :param start_point: 放大后的起点
        :param area_B:      放大后的区域
        """
        self.for_local_planner_max_limit = 2
        self.sample_times = 5e3
        self.moon_map = moon_map  # 用于可视化
        self.height = [[0 for _ in range(500)] for _ in range(500)]  # after *10 !
        self.start_pos = start_point  # in 500 * 500
        self.end_pos = [area_B[0], area_B[1]]  # 直接用 500 * 500 的坐标系
        self.area_B = area_B  # 放大后的地图
        self.xlength = area_B[2]
        self.ylength = area_B[3]
        self.gradient_limit = 1.6  # 梯度限制，用于可通行性检测 TODO change
        self.search_length = 5  # 10太大了，5应该正好

        """
        可通行性严格： self.gradient_limit = 1.3  # 梯度限制，用于可通行性检测 TODO change
                self.search_length = 7  # 10太大了，5应该正好
        松：          self.gradient_limit = 1.7
                self.search_length = 3 or 4
        """
        self.path = None
        for i in range(len(moon_map)):
            for j in range(len(moon_map[i])):
                self.height[i][j] = moon_map[i][j][3] * 10
        pre_search = {}
        ind = 1
        for i in range(1, len(self.height[0]) - 1):  # TODO bughere
            for j in range(1, len(self.height) - 1):
                # check 可通行性判断阶段
                dz = abs(self.height[j][i + 1] - self.height[j][i - 1]) / 2
                dz = max(abs(self.height[j - 1][i] - self.height[j + 1][i]) / 2, dz)
                if dz > self.gradient_limit and (not (i == self.start_pos[0] and j == self.start_pos[1])) and (
                not (i == self.end_pos[0] and j == self.end_pos[1])):  # 差距大于10cm
                    # plt.plot(i, j, 'bo', markersize=1)
                    continue

                if i not in pre_search:
                    pre_search[i] = {}
                pre_search[i][j] = ind  # for search
                ind += 1
        self.search = pre_search

    # 坐标系转换上可能有问题
    def in_target_area(self, node):  # rrt算法停止条件
        node = list(node)
        # node[0] = node[0] / 10
        # node[1] = node[1] / 10
        # node[0] = node[0] - 25
        # node[1] = node[1] - 25
        x = node[0]
        y = node[1]
        if self.area_B[0] - self.area_B[2] / 2 <= x <= self.area_B[0] + self.area_B[2] / 2 and self.area_B[
            1] - self.area_B[3] / 2 <= y <= self.area_B[1] + self.area_B[3] / 2:
            return True
        return False

    def heuristic(self, a, b):
        ca = self.height[a[1]][a[0]]  # 暂时用左下角点进行路径规划 TO change
        cb = self.height[b[1]][b[0]]  # TODO change
        return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2 + (ca - cb) ** 2)  # 空间距离

    def neighbors(self, pos):  # 改进搜索策略
        x, y = pos
        results = []
        for i in range(x - 1, x + 2):  # 搜索两维邻域
            for j in range(y - 1, y + 2):
                if i in self.search and j in self.search[i] and self.is_collision_free(None, [x - 1, y],
                                                                                       search_length=self.search_length):
                    results.append((i, j))
        return results

    def cost(self, a, b):
        ca = self.height[a[1]][a[0]]  # 暂时用左下角点进行路径规划 TO change
        cb = self.height[b[1]][b[0]]  # TODO change
        return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2 + 5 * (ca - cb) ** 2)  # 空间距离

    def twoDimDistance(self, node1, node2):
        return math.sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2)

    def get_nearest_node(self, random_node, nodes):
        min_distance = float('inf')
        nearest_node = None
        for node in nodes:
            d = self.twoDimDistance(node, random_node)
            if d < min_distance:
                min_distance = d
                nearest_node = node
        return nearest_node

    def get_new_node(self, nearest_node, random_node, max_limit):
        d = self.twoDimDistance(nearest_node, random_node)
        if d <= max_limit:
            return random_node
        else:
            theta = math.atan2(random_node[1] - nearest_node[1], random_node[0] - nearest_node[0])
            new_x = nearest_node[0] + max_limit * math.cos(theta)
            new_y = nearest_node[1] + max_limit * math.sin(theta)  # 最大扩展
            return (new_x, new_y)  # tuple

    # rrt算法时 可通行区域判断
    def is_collision_free(self, nearest_node, new_node, search_length):
        x = int(new_node[0])
        y = int(new_node[1])
        for i in range(x - search_length, x + search_length + 1):  # 搜索临近矩形区域
            for j in range(y - search_length, y + search_length + 1):
                if i not in self.search or j not in self.search[i]:
                    return False
        return True

    def astar(self, start, goal, visual=False, addition=False):  # TODO int only
        """
        目前astar局限在整数路径规划层面
        :param additon: 额外路径生成选项
        :return:
        """
        start = tuple((int(start[0]), int(start[1])))
        goal = tuple((int(goal[0]), int(goal[1])))

        frontier = PriorityQueue()
        frontier.put([0, start])
        came_from = {}
        cost_so_far = {}
        came_from[start] = None
        cost_so_far[start] = 0
        while not frontier.empty():
            priority, current = frontier.get()

            if current == goal:
                break

            for next in self.neighbors(current):  # 计算出一条完整路线
                new_cost = cost_so_far[current] + self.cost(current, next)
                if (next not in cost_so_far) or (new_cost < cost_so_far[next]):
                    cost_so_far[next] = new_cost
                    priority = new_cost + self.heuristic(goal, next)
                    frontier.put([priority, next])
                    came_from[next] = current

        if goal not in came_from:  # 未搜索到目标点
            # print("A* planner error")
            return None

        pth = []
        node = goal
        while node != start:
            pth.append(node)
            node = came_from[node]
        pth.append(start)
        pth.reverse()
        # 路径可视化
        if visual:
            plt.imshow(self.moon_map, cmap='gray')
            for node in pth:
                plt.plot(node[0], node[1], 'ro', markersize=1)  # plot还是正常,先横坐标再纵坐标
            plt.show()
        return pth  # path with two dimension

    def goal_bias_rrt(self, start, goal, addition=True, visual=False):
        start = tuple(start)
        goal = tuple(goal)
        iterations = self.sample_times
        nodes = []
        nodes.append(start)
        came_from = {}

        real_goal = None
        for i in range(int(iterations)):
            random_node = None
            if random.random() < 0.10:  # 0.1的几率Goal bias
                random_node = goal
            else:
                random_node = (random.randint(1, 500), random.randint(1, 500))

            nearest_node = self.get_nearest_node(random_node, nodes)
            new_node = self.get_new_node(nearest_node, random_node, self.for_local_planner_max_limit)

            if addition:  # 搜索额外路径 add version 2
                if self.is_collision_free(nearest_node, new_node, search_length=self.search_length):
                    if self.twoDimDistance(new_node, goal) < self.for_local_planner_max_limit * 2:
                        nodes.append(goal)
                        came_from[goal] = nearest_node
                        real_goal = goal
                        break
                    else:
                        came_from[new_node] = nearest_node
                        nodes.append(new_node)
            else:
                if self.is_collision_free(nearest_node, new_node,
                                          search_length=self.search_length):  # TODO taghere 扩大了不合法区域
                    if self.in_target_area(new_node):
                        nodes.append(new_node)  # here !!!
                        came_from[new_node] = nearest_node
                        real_goal = new_node
                        break
                    else:
                        came_from[new_node] = nearest_node
                        nodes.append(new_node)
        if real_goal == None:
            # print("goal-bias rrt planner error")
            return None
        path = []
        cur = real_goal
        path.append(real_goal)
        while cur in came_from:
            path.append(came_from[cur])
            cur = came_from[cur]
        path.reverse()
        # 路径可视化
        if visual:
            plt.imshow(self.moon_map, cmap='gray')
            for node in nodes:
                plt.plot(node[0], node[1], 'ro', markersize=1)
            for node in path:
                plt.plot(node[0], node[1], 'bo', markersize=1)
            plt.show()
        return path

    def rrt(self, start, goal, addition=True, visual=False):
        """
        :param visual:  可视化选项
        :param addition:额外路径生成时 设置为 True
        :return:
        """
        start = tuple(start)
        goal = tuple(goal)
        tolerance = 2
        iterations = self.sample_times
        nodes = []  # rrt tree
        nodes.append(start)
        came_from = {}
        real_goal = None
        for i in range(int(iterations)):  # 速度可以优化，eg get_nearest_node
            # random_node = (random.uniform(1, 500), random.uniform(1, 500))
            random_node = (random.randint(1, 500), random.randint(1, 500))
            nearest_node = self.get_nearest_node(random_node, nodes)
            new_node = self.get_new_node(nearest_node, random_node, self.for_local_planner_max_limit)
            if addition:  # 搜索额外路径 add version 2
                if self.is_collision_free(nearest_node, new_node, search_length=self.search_length):
                    if self.twoDimDistance(new_node, goal) < self.for_local_planner_max_limit * 2:
                        nodes.append(goal)
                        came_from[goal] = nearest_node
                        real_goal = goal
                        break
                    else:
                        came_from[new_node] = nearest_node
                        nodes.append(new_node)
            else:
                if self.is_collision_free(nearest_node, new_node, search_length=self.search_length):
                    if self.in_target_area(new_node):
                        nodes.append(new_node)
                        came_from[new_node] = nearest_node
                        real_goal = new_node
                        break
                    else:
                        came_from[new_node] = nearest_node
                        nodes.append(new_node)

        if real_goal == None:
            # print("rrt planner error")
            return None
        path = []
        cur = real_goal
        path.append(real_goal)
        while cur in came_from:
            path.append(came_from[cur])
            cur = came_from[cur]
        path.reverse()
        # 路径可视化
        if visual:
            plt.imshow(self.moon_map, cmap='gray')
            for node in nodes:
                plt.plot(node[0], node[1], 'ro', markersize=1)
            for node in path:
                plt.plot(node[0], node[1], 'bo', markersize=1)
            plt.show()
        return path

    def merged_rrt(self, start, goal, visual=False):
        """
        启发式规划
        通过两次goal bias rrt与两次rrt
        计算出最短的路径，返回最短路径
        """
        paths = []
        for i in range(2):
            path = self.goal_bias_rrt(tuple(start), tuple(goal), addition=True, visual=visual)
            if path != None:
                paths.append(path)
        for i in range(2):
            path = self.rrt(tuple(start), tuple(goal), addition=True, visual=visual)
            if path != None:
                paths.append(path)

        cur_min = float('inf')
        ret_path = None
        for path in paths:
            cur_dis = 0
            for j in range(len(path) - 1):
                cur_dis += self.twoDimDistance(path[j], path[j + 1])
            if cur_min > cur_dis:
                ret_path = path
                cur_min = cur_dis
        return ret_path

    # 终点策略板块(previous version
    def generate_additional_point(self, cur_point):  # add version 2 可以加一些随机性，终点策略代码
        limit = (self.area_B[2]) / 2  # 目标区域的长作为限制
        theta = math.atan2(cur_point[1] - self.start_pos[1], cur_point[0] - self.start_pos[0])  # atan2 (y,x)
        new_node = (cur_point[0] + limit * math.cos(theta), cur_point[1] + limit * math.sin(theta))
        while not self.in_target_area(new_node):
            limit /= 2
            new_node = (cur_point[0] + limit * math.cos(theta), cur_point[1] + limit * math.sin(theta))
        return new_node  # return tuple

    # 轨迹平滑板块 贝塞尔轨迹优化，来源 https://gitee.com/alanby/PathPlanning/tree/master
    # there is another option for path smoothing
    def smooth_path(self, WayPoints, sparse=3, imax_factor=2):
        """
        :param sparse:          原路径稀疏系数
        :param imax_factor:     imax的除数因子
        :return:                平滑后路径大小 1 / (sparse)
        注意后置参数随便定可能导致1/sparse大小无法保证
        """
        new_WayPoints = []
        for i in range(len(WayPoints)):
            if i % sparse == 0:
                new_WayPoints.append(WayPoints[i])
        controlPoints = self.WayPoints_to_ControlPoints(new_WayPoints)  # len = n / sparse
        samplePoints = len(new_WayPoints)  # n / sparse

        traj = []
        imax = floor(len(controlPoints) / imax_factor)  # n / sparse / imax
        for i in range(imax):
            for t in np.linspace(0, 1, samplePoints):  # Bezier parameter t
                traj.append(self.bezier(t, controlPoints[0 + i * 2: 3 + i * 2]))
        # n^2 / sparse / sparse / imax_factor
        path = np.array(traj)
        cur_len = len(path)

        real_sparse = int(sparse * cur_len / len(WayPoints))

        ret_path = []
        for i in range(len(path)):
            if i % real_sparse == 0:
                ret_path.append(path[i])

        return ret_path

    def WayPoints_to_ControlPoints(self, WayPoints):  # generate the control points according to the way points
        WayPointsSample = []
        sampling = 1
        for i in range(int(len(WayPoints) / sampling)):  # down sampling for the way points with 4 intervals
            WayPointsSample.append(WayPoints[i * sampling])
        WayPointsSample.append(WayPoints[len(WayPoints) - 1])  # add the start point to the way point
        # print(WayPointsSample[0])
        ControlPoints = []
        for i in range(len(WayPointsSample) - 1):
            cp = [(WayPointsSample[0 + i][0] + WayPointsSample[1 + i][0]) / 2,
                  (WayPointsSample[0 + i][1] + WayPointsSample[1 + i][1]) / 2]
            ControlPoints.append(cp)
            if i + 1 < len(WayPointsSample):
                ControlPoints.append(WayPointsSample[i + 1])
        return np.array(ControlPoints)

    def Comb(self, n, i, t):  # Bernstein polynomials
        return comb(n, i) * t ** i * (1 - t) ** (n - i)

    def bezier(self, t, controlPoints):  # Bezier equation
        n = len(controlPoints) - 1
        return np.sum([self.Comb(n, i, t) * controlPoints[i] for i in range(n + 1)], axis=0)  # Bezier curve

    # 得到终点策略的路径点
    # 得到路径列表，需要依次经过其中路径点
    def get_final_path_list(self, middle_flag):
        times = 100
        radius = min(self.xlength, self.ylength) / 2
        span = radius / times
        real_pts = {}
        if middle_flag:
            return [self.end_pos]

        for i in range(times, 0, -1):  # 需要反向遍历 times,-1,-1
            radius = i * span
            pts = self.get_points_from_circle(self.end_pos, radius)
            for j in range(len(pts)):
                pt1 = self.end_pos
                pt2 = pts[j]
                theta = math.atan2(pt2[1] - pt1[1], pt2[0] - pt1[0])
                length = int(self.twoDimDistance(pt1, pt2))
                flag = True
                for lg in range(0, length + 1):
                    x = pt1[0] + math.cos(theta) * lg
                    y = pt1[1] + math.sin(theta) * lg
                    if not self.is_collision_free(None, [x, y], self.search_length):
                        flag = False
                        break
                if flag:
                    if j not in real_pts:
                        real_pts[j] = pts[j]

            if len(real_pts) == 8 or i == 1:
                ret_real_pts = []
                sorted_keys = sorted(real_pts.keys())
                for key in sorted_keys:
                    # print(key)
                    value = real_pts[key]
                    ret_real_pts.append(value)

                ind = 0
                mn = float('inf')
                for num in range(len(ret_real_pts)):
                    if mn > self.twoDimDistance(self.start_pos, ret_real_pts[num]):
                        mn = self.twoDimDistance(self.start_pos, ret_real_pts[num])
                        ind = num
                pts = []
                # 逆时针
                for num in range(len(ret_real_pts), 0, -1):
                    pts.append(ret_real_pts[(ind + num) % len(ret_real_pts)])

                # 返回最终需要遍历的点列
                return pts

    # 直接计算八个交点
    def get_points_from_circle(self, center, radius):
        points = []
        angle_increment = 2 * math.pi / 8  # 圆心角间距为 45 度，即 360 度 / 8
        for i in range(8):
            angle = i * angle_increment
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            points.append((x, y))
        return points

    # 接入局部避障信息
    def local_planner_infomation(self, box):
        """
        :param box:是检测到障碍物边缘的最小外接四边形的四个顶点的像素坐标构成的二维nparray，
        想要取四个顶点的坐标可以使用下面的代码。如果未检测到障碍物边缘，则box的值为[[-1,-1]]
        :return:
        """
        # 删除对应点
        [x_s, x_b, y_s, y_b] = box  # x_small, x_big,y_small,y_big
        x_s = int(x_s)
        x_b = int(x_b)
        y_s = int(y_s)
        y_b = int(y_b)

        for x in range(x_s, x_b + 1):
            for y in range(y_s, y_b + 1):
                if x in self.search and y in self.search[x]:
                    del self.search[x][y]

    def with_astar_deemed_to_plan_path(self, start, goal):
        """ astar规划
        :param start:
        :param goal:
        :return:
        """
        pre = self.for_local_planner_max_limit
        path = self.astar(start, goal)  # remember to tuple
        if path is None:
            path = self.merged_rrt(start, goal)
        while path is None:
            self.for_local_planner_max_limit += pre
            path = self.rrt(start, goal)
        self.for_local_planner_max_limit = pre
        return self.smooth_path(path, sparse=3)

    def without_astar_deemed_to_plan_path(self, start, goal):
        """
        无长度保证，但是一般很快
        :param start:
        :param goal:
        :return:
        """
        pre = self.for_local_planner_max_limit
        path = self.goal_bias_rrt(start, goal)  # remember to tuple
        if path is None:
            path = self.merged_rrt(start, goal)
        while path is None:
            self.for_local_planner_max_limit += pre
            path = self.rrt(start, goal)
        self.for_local_planner_max_limit = pre
        return self.smooth_path(path, sparse=3)

"""
稀疏后路径说明

A* 两个路径点最多间隔 0.3-0.5 m

RRT 两个路径最多间隔 self.search_length * 3 = 0.6m - 0.8m

可能障碍物太小

但是有搜索领域，小问题
"""



"""

1.走到终点
2.局部避障


论文
"""
# TODO
# 接入终点策略 √
# 局部避障效果 √ 局部避障具体逻辑（是否持续调用
# 路径稀疏度 应该不太要紧，search_length存在 √
# 文档突出得分点
# 最后需要合理调整 self.search_length 与 self.for_local_planner()

# TODO 文献引用

# 避障调用逻辑
# 终点检测逻辑
# 重复如何避免

# 容忍误差
