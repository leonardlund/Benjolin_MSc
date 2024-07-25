import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio
import cupy as cp
import pickle
import zipfile


class BenjoDataset(Dataset):
    def __init__(self, data_dir, n_mfcc=13, features='mfcc-2d', device='cuda', mfcc_normalization=True, normalization=True):
        self.data_dir = data_dir
        self.device = device
        # self.feature_dict = feature_dict
        self.n_mfcc = n_mfcc
        self.normalization = normalization
        self.mfcc_normalization = mfcc_normalization
        self.npz = np.load(self.data_dir)
        self.data = torch.as_tensor(self.npz["features"], dtype=torch.float32)
        self.params = self.npz["params"]
        self.shape = self.data.shape
        # print(self.shape)
        self.normalize()

    def __len__(self):
        return  self.shape[0]

    def __getitem__(self, index):
        datapoint = self.data[index, :, :]
        # print(datapoint.shape)
        if self.mfcc_normalization:
            datapoint[:self.n_mfcc, :] /= self.n_mfcc
        return datapoint.flatten()
        
        
    def normalize(self):
        means = torch.mean(self.data, axis=0)
        std = torch.std(self.data, axis=0)
        if torch.count_nonzero(torch.isnan(means))>0:
            raise 'Nan encountered in data'
        self.data -= means
        self.data /= std

    def get_benjo_params(self, index):
        params_string = 'this is a placeholder string. TODO please remove'
        params_array = self.params[index, :]
        return params_array, params_string

