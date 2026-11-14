import numpy as np
import h5py

# 读取h5文件
filename = '/home/Conghui/TFG-DM-lagr/datasets/lagr/dataset/Lagr_u3c_diffusion.h5'
with h5py.File(filename, 'r') as hf:
    x_train = hf['train'][:]  # 假设数据集在h5文件中的键是'data'

# 提取前1024和后1024个数据
x_train = np.concatenate((x_train[:, :1024, :], x_train[:, -1024:, :]), axis=0)
x_train = np.concatenate((x_train[:, :, 0], x_train[:, :, 1],x_train[:, :, 2]), axis=0)
# 归一化处理
rx0, rx1 = np.amin(x_train), np.amax(x_train)
x_train = 2 * (x_train - rx0) / (rx1 - rx0) - 1

# 保存到新的h5文件
filename_out = '/home/Conghui/TFG-DM-lagr/datasets/lagr/dataset/Lagr_u3c_split_1024_diffusion.h5'
with h5py.File(filename_out, 'w') as hf:
    hf.create_dataset('min', data=rx0)
    hf.create_dataset('max', data=rx1)
    hf.create_dataset('train', data=x_train)
