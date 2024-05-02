import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio



class BenjoDataset(Dataset):
    def __init__(self, data_dir, features='mfcc-2d', device='cuda'):  # Adjust max_length as needed
        self.data_dir = data_dir
        self.features = features
        self.device = device
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".wav")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        waveform, sample_rate = torchaudio.load(path, normalize=True)

        if 'mfcc' in self.features:
            MFCC = torchaudio.transforms.MFCC(sample_rate=sample_rate,
                n_mfcc=13,
                melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False},)
            features = MFCC(waveform)
            if self.features == 'mfcc-bag-of-frames':
                mean = torch.mean(features, axis=0)
                std = torch.std(features, axis=0)
                feautrs = torch.concatenate(mean, std)

        return features

"""
if __name__ == "__main__":
    data_dir = ""
    dataset = BenjoDataset(data_dir, features='mfcc', device='cuda')
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=None, shuffle=True)
"""
