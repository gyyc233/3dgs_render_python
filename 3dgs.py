#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@ Description:
@ Date     : 2024/05/20 17:20:00
@ Author   : sunyifan
@ Version  : 1.0
"""

import math
import numpy as np
from tqdm import tqdm
from loguru import logger
from math import sqrt, ceil

from render_python import computeColorFromSH # 球谐函数计算颜色
from render_python import computeCov2D, computeCov3D # 计算协方差矩阵
from render_python import transformPoint4x4, in_frustum
from render_python import getWorld2View2, getProjectionMatrix, ndc2Pix, in_frustum


class Rasterizer:
    """
    光栅化
    """
    def __init__(self) -> None:
        pass

    def forward(
        self,
        P,  # int, num of guassians　高斯点数量
        D,  # int, degree of spherical harmonics 球谐函数阶数
        M,  # int, num of sh base function　球谐函数基函数数量
        background,  # color of background, default black 默认背景色
        width,  # int, width of output image 
        height,  # int, height of output image
        means3D,  # ()center position of 3d gaussian 每个高斯点的 3D 位置（中心坐标）
        shs,  # spherical harmonics coefficient 每个点对应的球谐系数
        colors_precomp, # 预计算的颜色
        opacities,  # opacities 每个高斯点的透明度
        scales,  # scale of 3d gaussians 每个高斯点的缩放尺度
        scale_modifier,  # default 1 缩放因子
        rotations,  # rotation of 3d gaussians 每个高斯点的旋转矩阵
        cov3d_precomp, # 预计算的协方差矩阵
        viewmatrix,  # matrix for view transformation 观测变换矩阵 世界到相机
        projmatrix,  # *(4, 4), matrix for transformation, aka mvp 投影矩阵（通常为 MVP = ModelViewProjection）
        cam_pos,  # position of camera 相机位置
        tan_fovx,  # float, tan value of fovx 水平视场角的正切值（用于投影计算）
        tan_fovy,  # float, tan value of fovy 垂直视场角的正切值
        prefiltered, # 是否已经做过预筛选

    ) -> None:
        """
        3D 高斯光栅化的主流程函数
        """

        # 根据 FOV 计算焦距
        focal_y = height / (2 * tan_fovy)  # focal of y axis
        focal_x = width / (2 * tan_fovx)

        # run preprocessing per-Gaussians
        # transformation, bounding, conversion of SHs to RGB
        # 3DGS预处理，判断是否在视野内，计算协方差矩阵，将球谐函数转为rgb等操作
        logger.info("Starting preprocess per 3d gaussian...")
        preprocessed = self.preprocess(
            P,
            D,
            M,
            means3D,
            scales,
            scale_modifier,
            rotations,
            opacities,
            shs,
            viewmatrix,
            projmatrix,
            cam_pos,
            width,
            height,
            focal_x,
            focal_y,
            tan_fovx,
            tan_fovy,
        )

        # produce [depth] key and corresponding guassian indices
        # sort indices by depth 高斯球按深度排序
        depths = preprocessed["depths"]
        print("depths:\n",depths)
        point_list = np.argsort(depths)
        print("sorted depths point_list:\n",point_list)

        # render
        # 对于每个像素，遍历所有高斯分布，根据它们的贡献（考虑透明度、距离等）进行累加，最后结合背景色得到最终像素颜色
        logger.info("Starting render...")
        out_color = self.render(
            point_list,
            width,
            height,
            preprocessed["points_xy_image"],
            preprocessed["rgbs"],
            preprocessed["conic_opacity"],
            background,
        )
        return out_color

    def preprocess(
        self,
        P,
        D,
        M,
        orig_points,
        scales,
        scale_modifier,
        rotations,
        opacities,
        shs,
        viewmatrix,
        projmatrix,
        cam_pos,
        W,
        H,
        focal_x,
        focal_y,
        tan_fovx,
        tan_fovy,
    ):
        """
        将 3D 高斯分布投影到 2D 屏幕上，并计算它们的视觉属性（如颜色、大小、透明度等）
        """

        rgbs = []  # rgb colors of gaussians 存储每个高斯分布的 RGB 颜色
        cov3Ds = []  # covariance of 3d gaussians 存储每个高斯分布的 3D 协方差矩阵
        depths = []  # depth of 3d gaussians after view&proj transformation 存储每个高斯分布在 NDC 空间[-1,1]中的深度值
        radii = []  # radius of 2d gaussians 存储每个高斯分布在屏幕空间中的半径
        conic_opacity = []  # covariance inverse of 2d gaussian and opacity 存储每个高斯分布的逆协方差矩阵和不透明度
        points_xy_image = []  # mean of 2d guassians 存储每个高斯分布在屏幕空间中的像素坐标

        for idx in range(P):
            # make sure point in frustum
            p_orig = orig_points[idx]
            # 观测变换，判断点是否在视锥体内
            p_view = in_frustum(p_orig, viewmatrix)
            if p_view is None:
                continue

            depths.append(p_view[2])
            print("depth:\n",p_view[2])

            # transform point, from world to ndc 将点从世界坐标转换到齐次裁剪空间 (Homogeneous Clip Space)
            # Notice, projmatrix already processed as mvp matrix
            p_hom = transformPoint4x4(p_orig, projmatrix)

            # 归一化齐次坐标的 z 分量,得到ndc [-1,1]坐标
            p_w = 1 / (p_hom[3] + 0.0000001)
            p_proj = [p_hom[0] * p_w, p_hom[1] * p_w, p_hom[2] * p_w]

            # compute 3d covarance by scaling and rotation parameters
            scale = scales[idx]
            rotation = rotations[idx]
            # 根据点的缩放旋转，计算3D高斯分布本身的协方差矩阵
            cov3D = computeCov3D(scale, scale_modifier, rotation)
            print("cov3Ds:\n",cov3D)
            cov3Ds.append(cov3D)

            # compute 2D screen-space covariance matrix
            # based on splatting, -> JW Sigma W^T J^T
            # 根据 3D 协方差矩阵和相机参数计算 2D 协方差矩阵
            cov = computeCov2D(
                p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix
            )

            # invert covarance(EWA splatting)
            det = cov[0] * cov[2] - cov[1] * cov[1] # 计算行列式
            # 如果行列式为 0，说明矩阵不可逆，跳过当前高斯分布
            if det == 0:
                depths.pop()
                cov3Ds.pop()
                continue

            # 计算逆协方差矩阵
            det_inv = 1 / det
            conic = [cov[2] * det_inv, -cov[1] * det_inv, cov[0] * det_inv]
            # 将逆协方差矩阵将与不透明度一起存入
            conic_opacity.append([conic[0], conic[1], conic[2], opacities[idx]])
            print("conic_opacity:\n",conic_opacity)

            # compute radius, by finding eigenvalues of 2d covariance 通过 2D 协方差矩阵的特征值计算高斯分布的半径
            # transfrom point from NDC to Pixel
            # 特征值代表了高斯椭圆在主轴方向上的“扩散程度”,最大特征值对应椭圆的最长轴长度
            mid = 0.5 * (cov[0] + cov[1])
            lambda1 = mid + sqrt(max(0.1, mid * mid - det))
            lambda2 = mid - sqrt(max(0.1, mid * mid - det))
            # 3*σ 表示在高斯分布中几乎覆盖全部能量(99.7%)
            my_radius = ceil(3 * sqrt(max(lambda1, lambda2)))
            print("my_radius:\n",my_radius)

            # 视口变换
            point_image = [ndc2Pix(p_proj[0], W), ndc2Pix(p_proj[1], H)]
            print("point_image:\n",point_image)

            radii.append(my_radius)
            points_xy_image.append(point_image)

            # convert spherical harmonics coefficients to RGB color
            # 将球谐函数系数转换为 RGB 颜色
            sh = shs[idx]
            result = computeColorFromSH(D, p_orig, cam_pos, sh)
            print("sh coef:\n",result)
            rgbs.append(result)

        return dict(
            rgbs=rgbs,
            cov3Ds=cov3Ds,
            depths=depths,
            radii=radii,
            conic_opacity=conic_opacity,
            points_xy_image=points_xy_image,
        )

    def render(
        self, point_list, W, H, points_xy_image, features, conic_opacity, bg_color
    ):
        """
        渲染函数
        point_list: 存储高斯分布的索引
        W: 屏幕宽度
        H: 屏幕高度
        points_xy_image: 存储高斯分布的像素坐标
        features: 存储高斯分布的球谐函数系数
        conic_opacity: 存储高斯分布的逆协方差矩阵和不透明度
        bg_color: 背景颜色
        """

        out_color = np.zeros((H, W, 3))
        # range(H * W) 创建一个从 0 到 H*W - 1 的迭代器，代表每一个像素
        # 使用 tqdm 库包装这个迭代器，创建一个带有进度条的可视化输出
        pbar = tqdm(range(H * W))

        # loop pixel
        for i in range(H):
            for j in range(W):
                pbar.update(1)
                pixf = [i, j]
                C = [0, 0, 0] # 初始化背景颜色

                # loop gaussian
                # 按深度顺序依次处理每个高斯分布
                for idx in point_list:

                    # init helper variables, transmirrance
                    T = 1

                    # Resample using conic matrix
                    # (cf. "Surface Splatting" by Zwicker et al., 2001)
                    # 计算2d高斯分布中心与正在遍历的像素的距离
                    xy = points_xy_image[idx]  # center of 2d gaussian
                    d = [
                        xy[0] - pixf[0],
                        xy[1] - pixf[1],
                    ]  # distance from center of pixel

                    # 根据 EWA Splatting 方法计算该2D高斯在当前像素处的指数衰减项 power
                    # 若大于 0 表示不在有效范围内，跳过
                    # 这里是2d高斯分布
                    con_o = conic_opacity[idx]
                    power = (
                        -0.5 * (con_o[0] * d[0] * d[0] + con_o[2] * d[1] * d[1])
                        - con_o[1] * d[0] * d[1]
                    )
                    # 交叉项感觉应该是正号
                    # power = (
                    #     -0.5 * (con_o[0] * d[0] * d[0] + con_o[2] * d[1] * d[1])
                    #     + con_o[1] * d[0] * d[1]
                    # )
                    if power > 0:
                        continue

                    # Eq. (2) from 3D Gaussian splatting paper.
                    # Compute color
                    # 计算2d高斯的不透明度 alpha
                    alpha = min(0.99, con_o[3] * np.exp(power))
                    if alpha < 1 / 255:
                        continue

                    # 如果剩余透明度 test_T 经过当前高斯后几乎完全遮挡，则提前跳出循环，后续高斯不可见
                    test_T = T * (1 - alpha)
                    if test_T < 0.0001:
                        break

                    # Eq. (3) from 3D Gaussian splatting paper.
                    # 将当前高斯的颜色乘以透明度 alpha 和当前累计透明因子 T，并叠加到当前像素颜色 C 上，更新剩余透明度 T
                    color = features[idx]
                    for ch in range(3):
                        C[ch] += color[ch] * alpha * T # T 没有被遮挡的概率，alpha 是2D高斯不透明度，color[ch] 是球谐函数模型颜色

                    T = test_T

                # get final color
                # 在所有高斯处理完成后，将未被遮挡的部分与背景颜色混合，得到最终像素颜色
                for ch in range(3):
                    out_color[j, i, ch] = C[ch] + T * bg_color[ch]

        return out_color


if __name__ == "__main__":
    # set guassian
    pts = np.array([[2, 0, -2], [0, 2, -1], [-2, 0, -3]]) # 定义了 3 个 3D 点，表示 3 个高斯分布的中心位置
    n = len(pts)
    shs = np.random.random((n, 16, 3)) # 初始化球谐函数系数
    opacities = np.ones((n, 1)) # 设置不透明度
    scales = np.ones((n, 3)) # 设置缩放系数
    rotations = np.array([np.eye(3)] * n) # 设置旋转函数，单位阵表示高斯分布没有遮挡

    # set camera
    cam_pos = np.array([0, 0, 5])
    R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]]) # 设置相机旋转
    # znear: 近裁剪平面距离; zfar: 远裁剪平面距离; fovX, fovY: 水平和垂直视场角
    proj_param = {"znear": 0.01, "zfar": 100, "fovX": 45, "fovY": 45}
    viewmatrix = getWorld2View2(R=R, t=cam_pos) # 观测变换矩阵
    print("view matrix:\n",viewmatrix)
    # ** 是一个字典解包操作符（Dictionary Unpacking Operator）
    # 将字典中的键值对作为关键字参数传递给函数
    projmatrix = getProjectionMatrix(**proj_param) # mvp投影矩阵
    print("project matrix:\n",projmatrix)

    projmatrix = np.dot(projmatrix, viewmatrix) # Model-View-Projection
    tanfovx = math.tan(proj_param["fovX"] * 0.5)
    tanfovy = math.tan(proj_param["fovY"] * 0.5)

    # render
    rasterizer = Rasterizer()
    out_color = rasterizer.forward(
        P=len(pts),
        D=3,
        M=16,
        background=np.array([0, 0, 0]),
        width=700,
        height=700,
        means3D=pts,
        shs=shs,
        colors_precomp=None,
        opacities=opacities,
        scales=scales,
        scale_modifier=1,
        rotations=rotations,
        cov3d_precomp=None,
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        cam_pos=cam_pos,
        tan_fovx=tanfovx,
        tan_fovy=tanfovy,
        prefiltered=None,
    )

    import matplotlib.pyplot as plt

    plt.imshow(out_color)
    plt.show()
