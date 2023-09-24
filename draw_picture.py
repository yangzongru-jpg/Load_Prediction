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

def origin_data_draw(origin_data_path,save_picture_path):
    """
    绘制时间序列真实值函数。

    参数:
    origin_data_path = '/home/yzr/msdis/data/Load_Data/0负荷数据202206/DG.xlsx'-->要完整的路径
    save_path = '/home/yzr/msdis/mycode/pictures/'
    """
    filename = os.path.basename(origin_data_path)
    print_name = os.path.splitext(filename)[0]

    # 使用pandas读取Excel文件
    df = pd.read_excel(origin_data_path)
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

def predict_data_draw(data_matrix,predict_data_matrix,save_path,name):
    """
    绘制时间序列真实值和预测值的函数。

    参数:
    data_matrix (torch.Tensor): 包含真实值的时间序列数据矩阵。
    predict_data_matrix (torch.Tensor): 包含预测值的时间序列数据矩阵。
    name (str): 时间序列的名字，将显示在图例中。
    """

    # 将 PyTorch Tensor 转换为 NumPy 数组
    data = data_matrix.cpu().numpy()
    predict_data = predict_data_matrix.cpu().numpy()

    # 创建 x 轴，假设时间序列是等间隔的，你可以根据实际情况修改这里
    x_gt_data = range(1, len(data) + 1)
    x_pred_data = range(len(data)+1-len(predict_data),len(data)+1)

    # 创建图形
    plt.figure(figsize=(10, 6))

    # 绘制真实值和预测值的曲线
    plt.plot(x_gt_data, data, label='True', linestyle='-', marker='o', markersize=5, color='blue')
    plt.plot(x_pred_data, predict_data, label='Predict', linestyle='-', marker='o', markersize=5, color='red')

    # 添加标题和标签
    plt.title(f'{name} Ground Truth and predict value')
    plt.xlabel('Timestamp')
    plt.ylabel('Value')

    # 添加图例
    plt.legend()

    # 显示图形
    plt.grid(True)
    plt.savefig(save_path+ '/'+name+'.jpg')
    plt.show()

#测试代码
# a = torch.randn(1000)
# b = torch.randn(500)
# origin_data_path = '/home/yzr/msdis/data/Load_Data/0负荷数据202206/DG.xlsx'
# save_path = '/home/yzr/msdis/mycode/pictures/'
# #predict_data_draw(a,b,'try',save_path)
# origin_data_draw(origin_data_path,save_path)