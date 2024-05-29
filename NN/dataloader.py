import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio


class BenjoDataset(Dataset):
    def __init__(self, data_dir, num_mfccs=90, features='mfcc-2d', device='cuda'):  # Adjust max_length as needed
        self.data_dir = data_dir
        self.features = features
        self.device = device
        self.num_mfccs = num_mfccs
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".wav")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]

        if self.features == 'params':
            params_array, _ = self.get_benjo_params(index)
            return torch.tensor(params_array, dtype=torch.float32).to(self.device) / 126

        waveform, sample_rate = torchaudio.load(path, normalize=True, format='wav')
        waveform = waveform[0, :].to(self.device)

        if 'mfcc' in self.features:
            MFCC = torchaudio.transforms.MFCC(sample_rate=sample_rate,
                                              n_mfcc=self.num_mfccs,
                                              melkwargs={"n_fft": 400, "hop_length": 160,
                                                         "n_mels": 101, "center": False}).to(self.device)
            features = MFCC(waveform)
            features = torch.clip(features, min=-25, max=25)
            shape = features.shape
            features += 25
            features /= 50
            if self.features == 'mfcc-bag-of-frames':
                mean = torch.mean(features, axis=1)
                std = torch.std(features, axis=1)
                features = torch.cat((mean, std))
                features = features.reshape((1, 2 * self.num_mfccs))
                return features
            else:
                features = features.reshape((1, shape[0], shape[1]))
                return features



    def get_benjo_params(self, index):
        path = self.files[index]
        params_string = path.removeprefix(self.data_dir + '/').removesuffix('.wav')
        params_list = params_string.split('-')
        params_array = np.array(list(map(int, params_list)))
        return params_array, params_string

