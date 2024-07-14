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
        self.feature_dict = feature_dict
        self.n_mfcc = n_mfcc
        self.weight_normalization = weight_normalization
        self.npz = np.load(self.data_dir)
        self.shape = self.data.shape
        self.data = cp.asarray(self.npz["features"])
        self.params = self.npz["params"]
        self.normalize()

    def __len__(self):
        return  self.shape[0]

    def __getitem__(self, index):
        datapoint = cp.array(self.data[index, :, :])
        if mfcc_normalization:
            datapoint[:, :self.n_mfcc, :] /= self.n_mfcc
        return torch.as_tensor(datapoint)
        
        
    def normalize(self):
        means = cp.mean(self.data, axis=0)
        std = cp.std(self.data, axis=0)
        self.data -= means
        self.data /= std

    def get_benjo_params(self, index):
        path = self.files[index]
        params_string = path.removeprefix(self.data_dir + '/').removesuffix('.wav')
        params_list = params_string.split('-')
        params_array = np.array(list(map(int, params_list)))
        return params_array, params_string

