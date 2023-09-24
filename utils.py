
import os
from typing import Any
import numpy as np
import pandas as pd
import glob
import re
import torch
from tqdm import tqdm
import sys
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings
from metrics import metric


def expand_hourly_data_to_15mins(original_matrix, samples_per_hour):
    """
    将1小时采样的矩阵扩展为15分钟采样的矩阵。

    参数：
    original_matrix (numpy.ndarray)：1小时采样的原始矩阵，形状为 (num_hours, num_features)。
    samples_per_hour (int)：每小时的采样次数，例如，若为4，则表示每小时采样4次。

    返回：
    numpy.ndarray：15分钟采样的矩阵，形状为 (num_hours * samples_per_hour, num_features)。
    """
    # 扩展后的矩阵形状
    new_shape = (original_matrix.shape[0] * samples_per_hour, original_matrix.shape[1])

    # 创建一个新的扩展矩阵
    expanded_matrix = np.zeros(new_shape)

    # 将原始矩阵的数据复制到扩展矩阵中
    for i in range(original_matrix.shape[0]):
        for j in range(samples_per_hour):
            expanded_matrix[i * samples_per_hour + j] = original_matrix[i]

    return expanded_matrix

def create_sliding_windows(all_matrix,window_size,predict_length):
    num_windows = len(all_matrix)-window_size-predict_length+1
    all_matrix = np.reshape(all_matrix,[len(all_matrix),-1])
    x_feature = torch.zeros(num_windows,window_size,all_matrix.shape[1])
    y_label = torch.zeros(num_windows,window_size)
    
    for i in range(num_windows):
        x_feature[i,:,:] = all_matrix[i:i+window_size,:]
        y_label[i,:] = all_matrix[i+window_size:i+window_size+predict_length,-1]
    return x_feature,y_label

def create_sliding_windows_for_day(all_matrix,window_size,forecast_horizon,step):
    # 定义训练集和测试集的分割比例
    train_ratio = 0.8
    test_ratio = 0.2
    # 计算用于训练和测试的样本数量
    total_samples = all_matrix.shape[0]
    train_samples = int(total_samples * train_ratio)
    test_samples = total_samples - train_samples
    # 创建训练集和测试集
    train_data = all_matrix[:train_samples]
    test_data = all_matrix[train_samples:]
    # 创建训练集和测试集的输入和输出序列
    x_train, y_train = create_sequences(train_data, window_size, step, forecast_horizon)
    x_test, y_test = create_sequences(test_data, window_size, step, forecast_horizon)
    return x_train,y_train,x_test,y_test
      
def create_sequences(data, window_size, step, forecast_horizon):
    x, y = [], []
    if data.dim() == 1:
        data = data.unsqueeze(1)
    for i in range(0, len(data) - window_size - forecast_horizon + 1, step):
        x.append(data[i:i + window_size, :])  # 输入特征，去除最后一列
        y.append(data[i + window_size:i + window_size + forecast_horizon, -1])  # 输出目标，后96个时间步的负荷
    
    return torch.stack(x), torch.stack(y)  # 直接返回 PyTorch 张量




def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    optimizer.zero_grad()

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        inputs,labels = data[0],data[1]
        pred = model(inputs.to(device))
        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()
        optimizer.step()
        optimizer.zero_grad()
    data_loader.desc = "[train epoch {}] loss: {:.3f}".format(epoch, accu_loss.item() / (step + 1))

    return accu_loss.item() / (step + 1)


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    data_loader = tqdm(data_loader, file=sys.stdout)
    pred_result=[]
    for step, data in enumerate(data_loader):
        inputs,labels = data[0].to(device),data[1].to(device)
        pred = model(inputs)
        pred_result.append(pred)
        metric_dict = metric(pred.to(device), labels)

    return pred_result,metric_dict

            
        
        








