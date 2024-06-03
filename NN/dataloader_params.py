import os
import numpy as np
import torch
from torch.utils.data import Dataset


class ParamXYDataset(Dataset):
    def __init__(self, data_dir, device='cuda'):  # Adjust max_length as needed
        self.data_dir = data_dir
        self.device = device
        self.file = np.load(data_dir)

    def __len__(self):
        return self.file.shape[0]

    def __getitem__(self, index):
        parameter_space = self.file[index]
        latent_space = self.file[8:]
        if parameter_space.shape[0] != 8:
            raise "Parameter space not 8-dimensional!"
        if latent_space.shape[0] != 2:
            raise "Latent space not 2-dimensional!"
        features = torch.tensor(parameter_space)
        target = torch.tensor(latent_space)
        return features, target
        