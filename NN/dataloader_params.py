import os
import numpy as np
import torch
from torch.utils.data import Dataset


class ParamXYDataset(Dataset):
    def __init__(self, data_dir, device='cuda'):  # Adjust max_length as needed
        self.data_dir = data_dir
        self.device = device
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npy")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        content = np.load(path)
        parameter_space = content[:8]
        latent_space = content[8:]
        if parameter_space.shape[0] != 8:
            raise "Parameter space not 8-dimensional!"
        if latent_space.shape[0] != 2:
            raise "Latent space not 2-dimensional!"
        features = torch.tensor(parameter_space)
        target = torch.tensor(latent_space)
        return features, target
        