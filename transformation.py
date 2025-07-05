#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@ Description:
@ Date     : 2024/05/17 11:13:25
@ Author   : sunyifan
@ Version  : 1.0
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# get (h, w, 3) cavas
# 创建一个形状为 (h, w, 3) 的 NumPy 数组，表示一个 RGB 图像画布
def create_canvas(h, w):
    return np.zeros((h, w, 3))


# 获取物体绕ｚ轴旋转指定角度的旋转矩阵
def get_model_matrix(angle):
    angle *= np.pi / 180
    return np.array(
        [
            [np.cos(angle), -np.sin(angle), 0, 0],
            [np.sin(angle), np.cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )


# from world to camera 相机放置在原点，相机坐标轴与世界坐标轴平行
def get_view_matrix(eye_pose):
    return np.array(
        [
            [1, 0, 0, -eye_pose[0]],
            [0, 1, 0, -eye_pose[1]],
            [0, 0, 1, -eye_pose[2]],
            [0, 0, 0, 1],
        ]
    )


# get projection, including perspective and orthographic 计算mvp投影矩阵
# fov 垂直视场角; aspect 屏幕宽高比（width / height）; near: 近裁剪平面距离; far: 远裁剪平面距离
# https://zhuanlan.zhihu.com/p/386900078
def get_proj_matrix(fov, aspect, near, far):
    t2a = np.tan(fov / 2.0) # 建垂直方向的缩放因子

    # 第一行控制 X 轴的缩放
    # 第二行控制 Y 轴的缩放
    # 第三行负责 Z 值映射到 [-1, 1] 区间
    # 第四行是标准的透视除法部分
    return np.array(
        [
            [1 / (aspect * t2a), 0, 0, 0],
            [0, 1 / t2a, 0, 0],
            [0, 0, (near + far) / (near - far), 2 * near * far / (near - far)],
            [0, 0, -1, 0],
        ]
    )

# 生成视口变换，将裁剪空间中的点映射到屏幕坐标系（像素坐标）
def get_viewport_matrix(h, w):
    # X 轴变换：将 NDC 坐标（范围 [-1, 1]）映射到 [0, w]
    # Y 轴变换：将 NDC 坐标（范围 [-1, 1]）映射到 [0, h]
    # Z 轴保持不变：保留深度值用于后续光栅化和深度测试
    return np.array(
        [[w / 2, 0, 0, w / 2], [0, h / 2, 0, h / 2], [0, 0, 1, 0], [0, 0, 0, 1]]
    )


if __name__ == "__main__":
    frame = create_canvas(700, 700)
    angle = 0
    eye = [0, 0, 5]
    pts = [[2, 0, -2], [0, 2, -2], [-2, 0, -2]]
    # 获取视口变换
    viewport = get_viewport_matrix(700, 700)

    # get mvp matrix
    # 观测变换
    mvp = get_model_matrix(angle) # 旋转
    mvp = np.dot(get_view_matrix(eye), mvp) # 平移
    # 4x4　投影变换（锥体到裁切空间clip space 立方体）
    # fov=45：垂直视角为45度，aspect=1：画面比例是正方形，near=0.1 和 far=50：表示可见深度范围从0.1到50
    mvp = np.dot(get_proj_matrix(45, 1, 0.1, 50), mvp)  

    # loop points
    pts_2d = []
    for p in pts:
        p = np.array(p + [1])  # 3x1 -> 4x1
        p = np.dot(mvp, p) # p=mvp*p 处于 裁剪空间（clip space） 的4维坐标 [x, y, z, w]
        p /= p[3] # 执行透视除法（perspective divide）,得到归一化设备坐标（NDC），范围在 [-1, 1] 之间

        # viewport
        p = np.dot(viewport, p)[:2] # 视口变换,只取 x 和 y 坐标,将NDC坐标映射到屏幕像素坐标
        pts_2d.append([int(p[0]), int(p[1])])

    vis = 1
    if vis:
        # visualize 3d
        fig = plt.figure()
        pts = np.array(pts)
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2] # 分别提取pts所有点的 x、y、z 坐标，作为3D散点图的数据

        ax = Axes3D(fig)
        ax.scatter(x, y, z, s=80, marker="^", c="g")
        ax.scatter([eye[0]], [eye[1]], [eye[2]], s=180, marker=7, c="r") # 绘制相机
        ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True, alpha=0.5) # 绘制这三点构成的三角面片（平面表面）
        plt.show()

        # visualize 2d
        c = (255, 255, 255)
        for i in range(3):
            for j in range(i + 1, 3):
                cv2.line(frame, pts_2d[i], pts_2d[j], c, 2)
        cv2.imshow("screen", frame)
        cv2.waitKey(0)
