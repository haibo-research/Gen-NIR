import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
sys.path.append('..')
sys.path.append('')
sys.path.append('../data')
sys.path.append('../data/SdfSamples')
import argparse
import signal
from tqdm import tqdm
from Utils import *
import bempp
from bempp.api.linalg import gmres



import torch
import numpy as np

def smooth_function_torch(vertices, faces, shape_derivative):
    num_vertices = len(vertices)
    device = shape_derivative.device  # 获取数据所在的设备，确保所有操作都在同一设备上进行

    # 确保faces是正步长的，通过复制数组，并转换为支持的类型
    faces_copy = np.array(faces, dtype=np.int64).copy()  # Convert to np.int64
    faces_torch = torch.tensor(faces_copy, dtype=torch.long, device=device)

    # 如果shape_derivative是Double类型，转换为Float
    if shape_derivative.dtype == torch.double:
        shape_derivative = shape_derivative.float()

    # 初始化每个顶点的函数值总和和邻居数量
    value_sum = torch.zeros(num_vertices, device=device, dtype=torch.float32)  # Specify dtype as float32
    neighbor_count = torch.zeros(num_vertices, device=device, dtype=torch.float32)  # Specify dtype as float32

    # 对每个面的每个顶点，更新相邻顶点的函数值总和和邻居数量
    for i in range(3):
        v1 = faces_torch[:, i]
        v2 = faces_torch[:, (i + 1) % 3]
        v3 = faces_torch[:, (i + 2) % 3]

        # 更新v1的邻居v2和v3
        value_sum.index_add_(0, v1, shape_derivative[v2] + shape_derivative[v3])
        # Create a tensor of 2s with the same size as v1 for index_add_
        twos = torch.full_like(v1, 2, dtype=torch.float32, device=device)  # Ensure dtype is float32
        neighbor_count.index_add_(0, v1, twos)

    # 计算平均值
    new_values = (value_sum + shape_derivative) / (neighbor_count + 1)

    return new_values









def integrate_over_surface_gaussian_vectorized(grid_vertices, grid_elements, value_vert, value_ele):
    # 高斯點和相應的權重
    gauss_points = np.array([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3], [1 / 3, 1 / 3]])
    gauss_weights = np.array([1 / 6, 1 / 6, 1 / 6, 1 / 6])

    # 提取所有三角形的頂點
    v1 = grid_vertices[:, grid_elements[0, :]]
    v2 = grid_vertices[:, grid_elements[1, :]]
    v3 = grid_vertices[:, grid_elements[2, :]]

    # 計算變換矩陣的雅可比行列式
    T = np.cross(v2 - v1, v3 - v1, axis=0)
    det_J = np.linalg.norm(T, axis=0) / 2

    # 初始化積分總和
    total_integral = 0.0

    # 對每個高斯點進行向量化操作
    for point, weight in zip(gauss_points, gauss_weights):
        # 高斯點在原始三角形中的坐標
        x = v1 + point[0] * (v2 - v1) + point[1] * (v3 - v1)

        if np.all(point == [1 / 3, 1 / 3]):  # 中心點
            function_values = value_ele
        else:
            # 插值計算在高斯點的函數值
            weights = np.array([1 - sum(point), point[0], point[1]])
            function_values = value_vert[grid_elements[0, :]] * weights[0] + \
                              value_vert[grid_elements[1, :]] * weights[1] + \
                              value_vert[grid_elements[2, :]] * weights[2]

        # 累加權重、函數值和雅可比行列式
        total_integral += weight * np.sum(function_values * det_J)

    return total_integral





def integrate_over_surface(grid_vertices, grid_elements, loss_coe):
    total_integral = 0.0
    for element in grid_elements.T:  # 遍历每个三角形元素
        # 获取三角形的顶点坐标
        v1, v2, v3 = grid_vertices[:, element[0]], grid_vertices[:, element[1]], grid_vertices[:, element[2]]

        # 计算三角形的面积
        area = np.linalg.norm(np.cross(v2 - v1, v3 - v1)) / 2

        # 计算函数在三角形顶点处的平均值
        avg_value = np.mean(loss_coe[element])

        # 累加到总积分
        total_integral += area * avg_value

    return total_integral