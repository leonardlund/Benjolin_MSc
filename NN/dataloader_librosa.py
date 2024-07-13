import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio
import cupy as cp
import pickle
import zipfile


class BenjoDataset(Dataset):
    def __init__(self, data_dir, features='mfcc-2d', device='cuda', weight_normalization=True):
        self.data_dir = data_dir
        self.device = device
        self.feature_dict = feature_dict
        self.weight_normalization = weight_normalization
        self.data = np.load(self.data_dir)
        self.shape = self.data.shape
        self.data = cp.asarray(self.data)

    def __len__(self):
        return  self.shape[0]

    def __getitem__(self, index):
        datapoint = self.data[index, :, :]
        


    def get_benjo_params(self, index):
        path = self.files[index]
        params_string = path.removeprefix(self.data_dir + '/').removesuffix('.wav')
        params_list = params_string.split('-')
        params_array = np.array(list(map(int, params_list)))
        return params_array, params_string

