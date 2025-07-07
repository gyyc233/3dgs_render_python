#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@ Description:
@ Date     : 2024/05/21 11:31:03
@ Author   : sunyifan
@ Version  : 1.0
"""

import numpy as np
from .graphic import getProjectionMatrix


def ndc2Pix(v, S):
    return ((v + 1.0) * S - 1.0) * 0.5


def in_frustum(p_orig, viewmatrix):
    # bring point to screen space
    p_view = transformPoint4x3(p_orig, viewmatrix)

    if p_view[2] <= 0.2:
        return None
    return p_view


def transformPoint4x4(p, matrix):
    # 下面是将matrix矩阵按照列优先顺序平展为一维数组
    matrix = np.array(matrix).flatten(order="F")
    x, y, z = p
    transformed = np.array(
        [
            matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12],
            matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13],
            matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14],
            matrix[3] * x + matrix[7] * y + matrix[11] * z + matrix[15],
        ]
    )
    return transformed


def transformPoint4x3(p, matrix):
    """
    将点进行rt变换
    """
    matrix = np.array(matrix).flatten(order="F")
    x, y, z = p
    transformed = np.array(
        [
            matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12],
            matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13],
            matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14],
        ]
    )
    return transformed


# covariance = RS[S^T][R^T]
def computeCov3D(scale, mod, rot):
    # create scaling matrix
    S = np.array(
        [[scale[0] * mod, 0, 0], [0, scale[1] * mod, 0], [0, 0, scale[2] * mod]]
    )

    # normalize quaternion to get valid rotation
    # we use rotation matrix
    R = rot

    # compute 3d world covariance matrix Sigma
    M = np.dot(R, S)
    cov3D = np.dot(M, M.T)

    return cov3D


def computeCov2D(mean, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix):
    # The following models the steps outlined by equations 29
    # and 31 in "EWA Splatting" (Zwicker et al., 2002).
    # Additionally considers aspect / scaling of viewport.
    # Transposes used to account for row-/column-major conventions.

    t = transformPoint4x3(mean, viewmatrix) # 点从世界坐标变换到相机坐标

    limx = 1.3 * tan_fovx
    limy = 1.3 * tan_fovy
    txtz = t[0] / t[2]
    tytz = t[1] / t[2]
    t[0] = min(limx, max(-limx, txtz)) * t[2]
    t[1] = min(limy, max(-limy, tytz)) * t[2]

    # 相机投影函数对相机空间坐标的导数
    J = np.array(
        [
            [focal_x / t[2], 0, -(focal_x * t[0]) / (t[2] * t[2])],
            [0, focal_y / t[2], -(focal_y * t[1]) / (t[2] * t[2])],
            [0, 0, 0],
        ]
    )

    print("J:\n",J)

    # 3D 协方差映射到屏幕空间
    W = viewmatrix[:3, :3]
    print("W:\n",W)
    T = np.dot(J, W)

    cov = np.dot(T, cov3D)
    cov = np.dot(cov, T.T)

    # Apply low-pass filter
    # Every Gaussia should be at least one pixel wide/high
    # Discard 3rd row and column
    # 低通滤波器（防锯齿）
    cov[0, 0] += 0.3
    cov[1, 1] += 0.3
    print("computeCov2D:\n",cov)

    # 返回xx,xy,yy 协方差矩阵是对称矩阵
    return [cov[0, 0], cov[0, 1], cov[1, 1]]


if __name__ == "__main__":
    p = [2, 0, -2]
    proj_param = {"znear": 0.01, "zfar": 100, "fovX": 45, "fovY": 45}
    projmatrix = getProjectionMatrix(**proj_param)
    transformed = transformPoint4x4(p, projmatrix)
    print(transformed)
