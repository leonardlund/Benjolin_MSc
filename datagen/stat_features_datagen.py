from NN.dataloader import BenjoDataset
from alive_progress import alive_bar
import torch
import os
import numpy as np

data_directory = os.path.normpath(r"/home/midml/Desktop/Leo_project/Benjolin_MA/audio")
data = BenjoDataset(data_directory, features="bag-of-frames", device="cuda", num_mfccs=24)
data_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True)

save_path = "/home/midml/Desktop/Leo_project/Benjolin_MA/bag-of-frames-dataset/"

huge_matrix = np.zeros((len(data), data[0].shape[0], data[0].shape[1]))
with (alive_bar(total=len(data)) as bar):
    for i, datapoint in enumerate(data_loader):
        arr, name = data.get_benjo_params(i)

        np.save(save_path + name + ".npy", datapoint)
        if i % 100 == 0:
            print(i)
        bar()
