# Autonomous-Navigation-and-Path-Planning-for-Deep-Space-Prospecting-Rovers
Autonomous Navigation and Path Planning for Deep Space Prospecting Rovers Project's code
in The 18th "Challenge Cup" Competition "Revealing the Leader" Special Contest programmed by team of Shanghai Jiao Tong University.
第十八届挑战杯竞赛“揭榜挂帅”赛道中由上海交通大学团队撰写的深空探矿巡视器的自主导航与路径规划项目代码。

## 背景:
### 题目介绍
地外天体探测存在严苛环境未知、先验知识欠缺和通信条
件恶劣等难题，目前已发射的月球和火星地表巡视器，其探测
任务多为地面完成路径规划后上注到巡视器上执行，任务执行
效率低，并且依赖高精度的星表地图。无地图陌生环境下的移
动机器人区域探索和目标搜寻是目前人工智能和无人系统领域
的前沿方向，如 2022 年 5 月发表于 Science Robotics 的封面文
章《Swarm of micro flying robots in the wild》展示了无人机集群
在未知复杂环境下的智能导航与快速避障。在未来的地外天体
探测任务中，为提高巡视器探测效率、降低对高精度地图依赖，
巡视器自主导航与路径规划是必然发展方向。因此，本题目以
地外天体探测巡视器为研究对象，期望解决无地图或仅有简易
地图条件下的巡视器自主导航与路径规划问题，提高巡视器在
复杂未知环境中的主动探索和主动适应能力，实现探测效能的
跨越式提升。


### 作品要求
本作品要求四轮巡视器在指定区域内，仅使用 RGBD 相机，
设计自主导航与路径规划算法，实现动态路径规划、自主导航
避障、兴趣目标搜索等功能。

## 项目内容
本github仓库用于保存第十八届挑战杯课外科技学术作品竞赛“揭榜挂帅”专项赛中上海交通大学深空探矿团队所撰写的代码。

### 项目简介
随着地球地表易开采矿业资源日益枯竭，而月球等地外天体上探明了丰富的矿
物与巨大的开采价值，太空采矿逐渐成为地外资源勘探的研究重点。本项目借鉴了
SLAM 的整体框架以及定位思路，设计了一套月球矿物开采任务驱动的巡视器自主
导航系统，仅使用 RGB-D 相机作为传感器完成了月球巡视器的太空矿产资源勘探任
务。
- 本项目针对月球退化环境搭建了一种定位框架，实现了较高精度的巡视器自定
位。
- 本项目结合多种路径规划算法可实现月球上的全局与局部路径规划，具有良好的
普适性和灵活性。
- 为确保巡视器的平稳行驶，本项目基于四轮差速模型设计了适应月
球环境的控制算法。
- 出于安全性的考虑，本项目设计了快速准确的避障算法，能基于
深度图准确识别障碍物并通过动态局部路径规划将其避开。
- 本项目还设计了两种兴
趣目标识别算法，具有较高的准确率。

为验证项目算法的可靠性和完备性，本项目在给定的 Webots 仿真环境中选择了多个不同的起点向目标区域导航，巡视器均能顺利
到达目的地并完成兴趣目标搜索任务。

### 算法方案设计示意图
![figure of algorithm design](/images/overall_image.png)

### 算法模块说明

#### 导航定位模块
![image](/images/nevigation.png)

### 路径规划模块
#### 全局路径规划
![image](/images/path_planning1.png)
#### 局部路径规划
![image](/images/path_planning2.png)
### 控制模块
![image](/images/control.png)

### 识别模块
#### 障碍物识别
![image](/images/recognition1.png)
#### 目标识别
![image](/images/recognition2.png)

