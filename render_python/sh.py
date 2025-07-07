#!/usr/bin/env python
# -*- encoding: utf-8 -*-

#  Copyright 2021 The PlenOctree Authors.
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.

"""
@ Description:
@ Date     : 2024/05/22 14:19:32
@ Author   : sunyifan
@ Version  : 1.0
"""

import numpy as np

# 球谐函数（Spherical Harmonics, SH）的归一化系数
# 用于将球谐系数转换为实际光照颜色。它们是预先计算好的常数

SH_C0 = 0.28209479177387814 # 常数项（环境光）
SH_C1 = 0.4886025119029199 # 线性项（方向光）分别代表 y, z, x 方向的线性光照变化

# 二次项（更精细的方向变化）用于重建更复杂的光照分布，比如漫反射表面的二阶光照效果
SH_C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
]

# 三次项（高阶细节）更高的阶数能表达更复杂的光照信息，但也会引入噪声或过拟合风险
SH_C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]


def computeColorFromSH(deg, pos, campos, sh):
    """
    Compute the color of a point from spherical harmonics coefficients
    :param deg: degree of the spherical harmonics 球谐函数的阶数(degree)，决定光照复杂度
    :param pos: 3D point position 3D 点的位置
    :param campos: camera position 相机位置，用于计算观察方向
    :param sh: spherical harmonics coefficients 球谐系数数组，每个通道对应 RGB
    :return: color of the point
    """
    # The implementation is loosely based on code for
    # "Differentiable Point-Based Radiance Fields for
    # Efficient View Synthesis" by Zhang et al. (2022)

    # 从相机指向当前点的方向向量，并归一化为单位向量
    dir = pos - campos
    dir = dir / np.linalg.norm(dir)

    # 0阶，对环境光的建模，与方向无关
    result = SH_C0 * sh[0]

    if deg > 0:
        x, y, z = dir
        # 1阶，对沿 x/y/z 方向的线性光照的建模，注意符号 - + -
        result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3]

        if deg > 1:
            # 2jie 构造二阶组合项
            xx = x * x
            yy = y * y
            zz = z * z
            xy = x * y
            yz = y * z
            xz = x * z
            result = (
                result
                + SH_C2[0] * xy * sh[4]
                + SH_C2[1] * yz * sh[5]
                + SH_C2[2] * (2.0 * zz - xx - yy) * sh[6]
                + SH_C2[3] * xz * sh[7]
                + SH_C2[4] * (xx - yy) * sh[8]
            )

            if deg > 2:
                # 3阶
                result = (
                    result
                    + SH_C3[0] * y * (3.0 * xx - yy) * sh[9]
                    + SH_C3[1] * xy * z * sh[10]
                    + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh[11]
                    + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh[12]
                    + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh[13]
                    + SH_C3[5] * z * (xx - yy) * sh[14]
                    + SH_C3[6] * x * (xx - 3.0 * yy) * sh[15]
                )
    
    # 加上 0.5 是为了补偿某些 SH 数据集的归一化偏移，另外值被限制在0-1
    result += 0.5

    return np.clip(result, a_min=0, a_max=1)


if __name__ == "__main__":
    deg = 3
    pos = np.array([2, 0, -2])
    campos = np.array([0, 0, 5])
    sh = np.random.random((16, 3))
    computeColorFromSH(deg, pos, campos, sh)
