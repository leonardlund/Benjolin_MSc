import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio
import cupy as cp
from dataloader import BenjoDataset
import pickle


if __name__ == '__main__':
    path = r"dir"
    feature_type = 'bag-of-frames'
    n_mfccs = 13
    feature_dict = {"mfcc": True, "centroid": True, "zcr": True, "rms": True, "flux": True, "flatness": True}
    windowing_args = {"win_length": 1024, "hop_size": 64, "pad": 0}
    weight_normalization = True

    data = BenjoDataset(path, features=feature_type, num_mfccs=n_mfccs, device="cuda",
                    fft_args=windowing_args, weight_normalization=weight_normalization,
                    feature_dict=feature_dict)

    cupy_arr = cp.zeros(shape=(len(data), data.shape[0], data.shape[1]))
    for i in range(len(data)):
        cupy_arr[i, :, :] = cp.asarray(data[i])

    mean = cp.asnumpy(cp.mean(cupy_arr, axis=0))
    std = cp.asnumpy(cp.std(cupy_arr, axis=0))
    
    stat_dictionary = {}
    stat_dictionary["mfcc-mean"] = mean[:13, :]
    stat_dictionary["mfcc-std"] = std[:13, :]
    stat_dictionary["centroid-mean"] = mean[13, :]
    stat_dictionary["centroid-std"] = std[13, :]
    stat_dictionary["zcr-mean"] = mean[14, :]
    stat_dictionary["zcr-std"] = std[14, :]
    stat_dictionary["rms-mean"] = mean[15, :]
    stat_dictionary["rms-std"] = std[15, :]
    stat_dictionary["flux-mean"] = mean[16, :]
    stat_dictionary["flux-std"] = std[16, :]
    stat_dictionary["flatness-mean"] = mean[17, :]
    stat_dictionary["flatness-std"] = std[17, :]

    file_name = open(f"stat_dictionary{windowing_args['win_length']}-{windowing_args['hop_size']}.pkl","wb")
    pickle.dump(stat_dictionary, file_name)
    file_name.close()
