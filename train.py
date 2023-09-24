import pandas as pd
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from utils import expand_hourly_data_to_15mins,create_sliding_windows
from mymodel import LSTMModel
from torch.utils.data import DataLoader
import argparse


# 设置随机种子以保持可重复性
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
#超参数设置
# window_size = 96
# predict_length = 96
# input_size = 4
# hidden_size = 256
# num_layers = 2
# num_epochs = 1
# learning_rate = 0.01
# batch_size = 2048
# # 定义Excel文件的路径
# excel_file_path_weather = "/home/yzr/msdis/data/Load_Data/weather20170101-20230716.xlsx"
# excel_file_path_load = "/home/yzr/msdis/data/Load_Data/全网负荷20170101-20230716.xlsx"
# save_picture_path = "/home/yzr/msdis/mycode/pictures"
# 使用pandas读取Excel文件的第2、3和4列
df_weather = pd.read_excel(excel_file_path_weather, usecols=[1, 2, 3])
#df_weather = df_weather.values[:43824,:] #2017-2021.12.31-->对应的2021123123:00的数据
df_weather = torch.tensor(expand_hourly_data_to_15mins(df_weather.values[:43824,:],samples_per_hour=4),dtype=torch.float32) #扩展15分钟的采样,output shape (175296, 3)
df_load = pd.read_excel(excel_file_path_load,usecols = [1,2])
df_load_family = torch.tensor(df_load.values[:175296,0],dtype=torch.float32)#(175296,)
all_matrix = torch.cat((df_weather, df_load_family.reshape(-1, 1)), dim=1)
X,Y = create_sliding_windows(all_matrix,window_size=96, predict_length=96)
#设置train test数据集
train_ratio = 0.8
num_samples = all_matrix.shape[0]
train_size = train_ratio*num_samples
y_train = Y[:train_size,:]
x_test = X[train_size:,:,:]
y_test = Y[train_size:,:]
x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
#设置模型
model = LSTMModel(input_size, hidden_size, num_layers)
# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    for inputs,labels in train_loader:
        outputs = model(inputs)
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 模型评估
model.eval()
y_test_pred = []
with torch.no_grad():
    for inputs,labels in test_loader:
        pred = model(inputs)
        y_test_pred.append(pred)

# 将预测结果保存为图片
plt.figure(figsize=(12, 6))
plt.plot(all_matrix[:,3], label='True', marker='o', linestyle='-')
plt.plot(y_test_pred.numpy(), label='Predicted', marker='o', linestyle='--')
plt.legend()
plt.title(f'Time Series {i + 1} Prediction')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.savefig(f'/home/yzr/msdis/mycode/pictures/Time_Series.jpg')
plt.close()

print("预测图片已保存到指定路径下。")




'''
#画load的图
plt.figure(figsize=(12, 6))
plt.plot(data_matrix_load[:, 0], label="菏泽负荷用电")
plt.xlabel("Time")
plt.ylabel("Load")
plt.legend()
plt.title("Load Curve")
plt.grid(True)
output_path = save_picture_path+"/load_curve.jpg"
plt.savefig(output_path)
plt.show()
#draw feature picture
plt.figure(figsize=(12, 6))
plt.plot(data_matrix_weather[:, 0], label="humidity")
plt.plot(data_matrix_weather[:, 1], label="temperature")
plt.plot(data_matrix_weather[:, 2], label="code")
plt.xlabel("Time")
plt.ylabel("Load")
plt.legend()
plt.title("Load Curve")
plt.grid(True)
output_path = save_picture_path+"/feature.jpg"
plt.savefig(output_path)
plt.show()
'''

