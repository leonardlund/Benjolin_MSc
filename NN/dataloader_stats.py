import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio
import librosa.feature
import librosa
import scipy.stats as stats


class BenjoDatasetStats(Dataset):
    def __init__(self, data_dir, num_mfccs=25, features='mfcc-2d', device='cuda'):  # Adjust max_length as needed
        self.data_dir = data_dir
        self.features = features
        self.device = device
        self.num_mfccs = num_mfccs
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npy")]
        self.min = np.load("/home/midml/Desktop/Leo_project/Benjolin_MA/statstats/min.npy")
        self.max = np.load("/home/midml/Desktop/Leo_project/Benjolin_MA/statstats/max.npy")
        self.mean = np.load("/home/midml/Desktop/Leo_project/Benjolin_MA/statstats/mean.npy")
        self.std = np.load("/home/midml/Desktop/Leo_project/Benjolin_MA/statstats/std.npy")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]

        if self.features == 'params':
            params_array, _ = self.get_benjo_params(index)
            return torch.tensor(params_array, dtype=torch.float32).to(self.device) / 126

        if 'bag-of-frames' == self.features:
            features = np.load(path)
            features -= self.min
            features /= (self.max - self.min)
            return torch.tensor(features, dtype=torch.float32).to(self.device).flatten()

    def get_benjo_params(self, index):
        path = self.files[index]
        params_string = path.removeprefix(self.data_dir + '/').removesuffix('.npy')
        params_list = params_string.split('-')
        params_array = np.array(list(map(int, params_list)))
        return params_array, params_string


if __name__ == '__main__':
    path = "/home/midml/Desktop/Leo_project/Benjolin_MA/bag-of-frames-dataset/"
    files = os.listdir(path)
    arr = np.zeros(shape=(len(files), 36, 4))
    for i in range(len(files)):
        arr[i, :, :] = np.load(path + files[i])
        if i % 10000 == 0:
            print(i)

    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    minimum = arr.min(axis=0)
    maximum = arr.max(axis=0)
    print("Means: ", list(mean))
    print("Std: ", list(std))
    np.save("/home/midml/Desktop/Leo_project/Benjolin_MA/statstats/min.npy", minimum)
    np.save("/home/midml/Desktop/Leo_project/Benjolin_MA/statstats/max.npy", maximum)
    np.save("/home/midml/Desktop/Leo_project/Benjolin_MA/statstats/mean.npy", minimum)
    np.save("/home/midml/Desktop/Leo_project/Benjolin_MA/statstats/std.npy", maximum)
