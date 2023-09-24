import numpy as np
import torch

def RSE(pred, true):
    rse = torch.sqrt(torch.sum((true - pred) ** 2)) / torch.sqrt(torch.sum((true - true.mean()) ** 2))
    return rse


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = torch.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    mae = torch.abs(pred - true).mean()
    return mae


def MSE(pred, true):
    return torch.nn.functional.mse_loss(pred, true)


def RMSE(pred, true):
    return torch.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return torch.mean(torch.abs((pred - true) / true)) * 100


def MSPE(pred, true):
    return torch.mean(torch.pow((pred -true) / true, 2)) * 100


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    
    metric_dict = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'MSPE': mspe
    }

    return metric_dict