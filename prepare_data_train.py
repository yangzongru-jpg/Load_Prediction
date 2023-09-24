import os
from typing import Any
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings

def create_sliding_windows(all_matrix,window_size,predict_length):
    num_windows = len(all_matrix)-window_size-predict_length-1
    x_feature = torch.zeros(num_windows,window_size,all_matrix.shape[1])
    y_label = torch.zeros(num_windows,window_size)
    
    for i in range(num_windows):
        x_feature[i,:,:] = all_matrix[i:i+window_size,:]
        y_label[i,:] = all_matrix[i+window_size:i+window_size+predict_length,3]
    return x_feature,y_label