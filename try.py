import pandas as pd
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from utils import expand_hourly_data_to_15mins,create_sliding_windows
from mymodel import LSTMModel
from torch.utils.data import DataLoader
import argparse
import os
##################################################################################################################
#设置GPU和tensorboard
gpu_id = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
##################################################################################################################
excel_file_path = '/home/yzr/msdis/data/Load_Data/0负荷数据202206/CZ.xlsx'
save_picture_path = "/home/yzr/msdis/mycode/pictures"
folder_path = '/home/yzr/msdis/data/Load_Data/0负荷数据202206'

for filename in os.listdir(folder_path):
    if filename.endswith('.xlsx'):
        print_name = os.path.splitext(filename)[0]
        # 构建完整的文件路径
        file_path = os.path.join(folder_path, filename)
        # 使用pandas读取Excel文件
        df = pd.read_excel(file_path)
        # 将数据转换为PyTorch张量
        data_df = df.iloc[1:,1:]
        data_tensor = torch.tensor(data_df.values, dtype=torch.float32)
        data_tensor = data_tensor.view(-1)
        # 打印数据的形状
        print(f"{print_name}数据形状:", data_tensor.shape)
        #画图代码
        plt.figure(figsize=(12, 6))
        plt.plot(data_tensor, label=f"{print_name}负荷用电")
        plt.xlabel("Time")
        plt.ylabel("Load")
        plt.legend()
        plt.title(f"{print_name}Load Curve")
        plt.grid(True)
        image_filename = os.path.splitext(filename)[0] + '.jpg'
        output_path = os.path.join(save_picture_path, image_filename)
        plt.savefig(output_path)
        plt.show()
        print(f"{print_name} 处理完成")