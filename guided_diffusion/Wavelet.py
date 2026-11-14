import torch
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt


def multi_level_wavelet(x, signal_length=1024, levels=3, wavelet='haar'):
    if wavelet == 'haar':
        low = torch.tensor([1.0, 1.0]) / np.sqrt(2)
        high = torch.tensor([-1.0, 1.0]) / np.sqrt(2)
    else:
        raise NotImplementedError("仅支持Haar小波示例")

    def build_single_level_matrix(length):
        if length < 2:
            return None
        W = torch.zeros((length, length))
        for i in range(length // 2):
            W[i, 2 * i] = low[0]
            W[i, 2 * i + 1] = low[1]
            W[length // 2 + i, 2 * i] = high[0]
            W[length // 2 + i, 2 * i + 1] = high[1]
        return W

    def build_multilevel_matrix():
        matrices = []
        current_length = signal_length
        for _ in range(levels):
            if current_length < 2:
                break
            W = build_single_level_matrix(current_length)
            matrices.append(W)
            current_length = current_length // 2
        return matrices

    transform_matrices = build_multilevel_matrix()

    coefficients = []
    current_signal = x
    for level in range(levels):
        if current_signal.shape[-1] < 2 or level >= len(transform_matrices):
            break
        W = transform_matrices[level].to(x.device)
        transformed = torch.matmul(current_signal, W.T)
        approx = transformed[..., :transformed.shape[-1] // 2]
        detail = transformed[..., transformed.shape[-1] // 2:]
        coefficients.append(detail)
        current_signal = approx
    coefficients.append(current_signal)  # 最终近似系数

    # 构造矩阵形式
    matrix_form = torch.zeros_like(x)
    current_pos = 0

    # 填充近似系数
    approx = coefficients[-1]
    approx_length = approx.shape[-1]
    matrix_form[..., :approx_length] = approx
    current_pos += approx_length

    # 逆序填充细节系数
    for detail in reversed(coefficients[:-1]):
        detail_length = detail.shape[-1]
        end_pos = current_pos + detail_length
        matrix_form[..., current_pos:end_pos] = detail
        current_pos = end_pos

    return matrix_form


def wavelet_new(x, level):
    # 动态获取输入形状[64, 3, 1024]
    batch_size, features, seq_len = x.size()
    # 小波分解级数对应分组大小
    group_size = 2**level
    num_groups = seq_len // group_size  # 计算完整分组的数量
    # 分组并计算均值
    x_mean = torch.mean(x.view(batch_size,features, num_groups, group_size), dim=3)
    # 归一化使能量守恒
    x_mean = x_mean*group_size**(1/2)
    # x_mean = x_mean * 2 ** (level / 2)
    # # 填充0保持原始长度
    # padding_len = seq_len - num_groups  # 需要填充的长度
    # zeros = torch.zeros(batch_size, features, padding_len, device=x.device)
    # x_padded = torch.cat([x_mean, zeros], dim=2)

    return x_mean
