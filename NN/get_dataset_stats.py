
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio
import cupy as cp
from dataloader import BenjoDataset
import pickle


if __name__ == '__main__':
    path = r"/cephyr/users/lundle/Alvis/benjo/audio.zip"
    feature_type = 'bag-of-frames'
    n_mfccs = 13
    feature_dict = {"mfcc": True, "centroid": True, "zcr": True, "rms": True, "flux": True, "flatness": True}
    windowing_args = {"win_length": 1024, "hop_size": 64, "pad": 0}
    weight_normalization = False
    print("Is cuda available? ", torch.cuda.is_available())

    savepath = "/cephyr/users/lundle/Alvis/benjo/alvis_dataset_short_window_1.npz"

    data = BenjoDataset(path, features=feature_type, num_mfccs=n_mfccs, device="cuda",
                    fft_args=windowing_args, weight_normalization=weight_normalization,
                    feature_dict=feature_dict)

    torch_arr = torch.zeros((len(data), data[0].shape[0], data[0].shape[1]))
    params = np.zeros((len(data), 8))
    for i in range(len(data)):
        arr = data[i]
        print(i)
        if arr == None:
            continue
        torch_arr[i, :, :] = arr
        params[i, :], _ = data.get_benjo_params(i)
        if i % 1000 == 0:
            print(i)
            np.savez_compressed(savepath, features=torch_arr.cpu().numpy(), params=params)

            
    # mean = torch.nanmean(torch_arr, axis=0).cpu().numpy()
    # std = torch.std(torch_arr, axis=0).cpu().numpy()
    # print(mean, std)
    """
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

    file_name = open(f"/cephyr/users/lundle/Alvis/benjo/stat_dictionary{windowing_args['win_length']}-{windowing_args['hop_size']}.pkl","wb")
    pickle.dump(stat_dictionary, file_name)
    file_name.close()
    """
    np.savez_compressed(savepath, features=torch_arr.cpu().numpy(), params=params)
