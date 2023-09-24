import pandas as pd
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import math
import torch.optim.lr_scheduler as lr_scheduler
from utils import expand_hourly_data_to_15mins,create_sliding_windows
from utils import create_sliding_windows_for_day,train_one_epoch,evaluate
from mymodel import LSTMModel,GRUModel,RNNModel
from torch.utils.data import DataLoader
import os
import torch.optim as optim
import argparse
from draw_picture import origin_data_draw,predict_data_draw
import sys
sys.argv=['']
del sys


def get_argparser():
    parser = argparse.ArgumentParser(description='Times series under extrem condition')
    #设置gpu环境
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    #Dataset options
    parser.add_argument('--dataset', type=str, default='SD',
                        choices=['SD', 'GD'], help='Name of dataset')
    #模型超参数
    parser.add_argument('--model', default='lstm', choices=['lstm', 'gru', 'RNN'])
    parser.add_argument('--input_size', default=4, type=int, help='LSTM input size')
    parser.add_argument('--hidden_size', default=256, type=int, help='LSTM hidden size')
    parser.add_argument('--num_layers', default=2, type=int, help='LSTM num layers')
    parser.add_argument('--epochs', default=1, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--batch_size', default=2048, type=int, help='ts batch size')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate for KD')
    parser.add_argument('--window_size', default=96, type=int, help='ts window size')
    parser.add_argument('--forecast_horizon', default=96, type=int, help='predict_length')
    parser.add_argument('--step', type=int, default=96,help='模型每一次往前走的步长')
    parser.add_argument('--lrf', type=float, default=0.01,help='衰减学习率的参数')

    # 定义Excel文件的路径
    parser.add_argument('--SD_excel_file_path_weather', default='/home/yzr/msdis/data/Load_Data/weather20170101-20230716.xlsx', type=str,help='山东weather的路径')
    parser.add_argument('--SD_excel_file_path_load',default='/home/yzr/msdis/data/Load_Data/全网负荷20170101-20230716.xlsx',type=str, help='山东负荷数据路径')
    parser.add_argument('--save_picture_path',default='/home/yzr/msdis/Load_Prediction/pictures',type=str, help='保存图片路径')
    return parser

def get_dataset(opts):
    if opts.dataset == 'SD':
        origin_data_dict = {}
        excel_file_path_weather = "/home/yzr/msdis/data/Load_Data/weather20170101-20230716.xlsx"
        excel_file_path_load = "/home/yzr/msdis/data/Load_Data/全网负荷20170101-20230716.xlsx"
        # 使用pandas读取Excel文件的第2、3和4列
        df_weather = pd.read_excel(excel_file_path_weather, usecols=[1, 2, 3])
        #df_weather = df_weather.values[:43824,:] #2017-2021.12.31-->对应的2021123123:00的数据
        df_weather = torch.tensor(expand_hourly_data_to_15mins(df_weather.values[:43824,:],samples_per_hour=4),dtype=torch.float32) #扩展15分钟的采样,output shape (175296, 3)
        df_load = pd.read_excel(excel_file_path_load,usecols = [1,2])
        df_load_family = torch.tensor(df_load.values[:175296,0],dtype=torch.float32)#(175296,)
        all_matrix = torch.cat((df_weather, df_load_family.reshape(-1, 1)), dim=1)
        origin_data_dict['SD'] = all_matrix[:,-1]
        data_tensor = all_matrix
        x_train,y_train,x_test,y_test = create_sliding_windows_for_day(data_tensor,window_size=opts.window_size, forecast_horizon=opts.forecast_horizon,step = opts.step)
        train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
        test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True,num_workers=8)
        test_loader = DataLoader(test_dataset, batch_size=opts.batch_size,shuffle=False,num_workers=8)
        data_loaders_dict ={'train':train_loader,'test':test_loader}
        return data_loaders_dict,origin_data_dict
    elif opts.dataset == 'GD':
        folder_path = '/home/yzr/msdis/data/Load_Data/0负荷数据202206'
        batch_size = opts.batch_size
        GD_load_dict = {}
        origin_data_dict = {}
        keys = []
        for filename in os.listdir(folder_path):
            if filename.endswith('.xlsx'):
                city_name = os.path.splitext(filename)[0]
                keys.append(city_name)
                # 构建完整的文件路径
                file_path = os.path.join(folder_path, filename)
                # 使用pandas读取Excel文件
                df = pd.read_excel(file_path)
                # 将数据转换为PyTorch张量
                data_df = df.iloc[1:,1:]
                data_tensor = torch.tensor(data_df.values, dtype=torch.float32)
                data_tensor = data_tensor.view(-1)
                origin_data_dict[city_name]=data_tensor
                x_train,y_train,x_test,y_test = create_sliding_windows_for_day(data_tensor,window_size=opts.window_size, forecast_horizon=opts.forecast_horizon,step = opts.step)
                train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
                test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=8)
                test_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=False,num_workers=8)
                data_loaders_dict ={'train':train_loader,'test':test_loader}
                GD_load_dict[city_name]= data_loaders_dict
        return GD_load_dict,origin_data_dict
    
    


def main():
    opts = get_argparser().parse_args() 
    device = torch.device(opts.device if torch.cuda.is_available() else "cpu")
    data_loaders_dict,origin_data_dict = get_dataset(opts) #data_loaders_dict是这样调用的 data_loaders_dict['JM']['train']
    if opts.model == 'lstm':
        model = LSTMModel(input_size=96, hidden_size=256, num_layers=4).to(device)
    elif opts.model == 'gru':
        model = GRUModel(input_size=96, hidden_size=256, num_layers=2).to(device)
    elif opts.model == 'rnn':
        model = RNNModel(input_size=96, hidden_size=256, num_layers=2).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    lf = lambda x: ((1 + math.cos(x * math.pi / opts.epochs)) / 2) * (1 - opts.lrf) + opts.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    if opts.dataset == 'SD':
        train_loader = data_loaders_dict['train']
        test_loader = data_loaders_dict['test']
        for epoch in enumerate(range(opts.epochs)):
             # train
            train_loss = train_one_epoch(model=model,
                                            optimizer=optimizer,
                                            data_loader=train_loader,
                                            device=device,
                                            epoch=epoch)

            scheduler.step()

            # test
            pred_result,metric_dict = evaluate(model=model,
                                        data_loader=test_loader,
                                        device=device)
        print("SD train and test process is finish")
        #concat_tensor = pred_result[0].view(-1)
        pred_result_value = pred_result[0].view(-1)
        origin_data = origin_data_dict[opts.dataset]
        save_picture_path = "/home/yzr/msdis/Load_Prediction/pictures"
        predict_data_draw(data_matrix = origin_data,predict_data_matrix = pred_result_value,save_path = save_picture_path,name = 'SD')
        print("Draw GD predict picture is finish")
    elif opts.dataset == 'GD':
        for keys,values in data_loaders_dict.items():
            train_loader = data_loaders_dict[keys]['train']
            test_loader = data_loaders_dict[keys]['test']
            for epoch in enumerate(range(opts.epochs)):
                # train
                train_loss = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

                scheduler.step()

                # test
                pred_result,metric_dict = evaluate(model=model,
                                            data_loader=test_loader,
                                            device=device)
            print(f"{keys} train and test process is finish")
            #concat_tensor = pred_result[0].view(-1)
            pred_result_value = pred_result[0].view(-1)
            # for i in range(1,len(pred_result[0])):
            #     pred_result_value = torch.cat([concat_tensor, pred_result[i]], dim=0)
            origin_data = origin_data_dict[opts.dataset]
            save_picture_path = "/home/yzr/msdis/Load_Prediction/pictures"
            predict_data_draw(data_matrix = origin_data,predict_data_matrix = pred_result_value,save_path = save_picture_path,name = keys)
            print(f"Draw {keys} predict picture is finish")
if __name__ == '__main__':
    main()