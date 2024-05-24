
from VAE import *
from dataloader import *
from train import *
import numpy as np
import os
import matplotlib.pyplot as plt
from plot import plot_mfcc_spectrograms_side_by_side
import random
from alive_progress import alive_bar


data_directory = os.path.normpath(r"/home/midml/Desktop/Leo_project/Benjolin_MA/audio")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: ", device)

bag_of_frames = True

feature_type = 'mfcc-bag-of-frames' if bag_of_frames else 'mfcc-2d'
data = BenjoDataset(data_directory, features=feature_type, device=device, num_mfccs=40)
data_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True)

input_shape = data[0].shape
print(input_shape)
input_dim = 80 if bag_of_frames else input_shape[1] * input_shape[2]
hidden_dim = input_dim // 2
latent_dim = 16

vae = VAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)  # batch_size=32

vae = vae.to(device)
model_directory = "/home/midml/Desktop/Leo_project/Benjolin_MA/NN/models/test-bag-beta1e-4-40mfcc"
vae.load_state_dict(torch.load(model_directory))
print("Loaded model from ", model_directory, " successfully!")
plt.rcParams['figure.dpi'] = 150

parameter_matrix = np.zeros((len(data), 8))
latent_matrix = np.zeros((len(data), 16))

with (alive_bar(total=len(data)) as bar):
    for i, datapoint in enumerate(data_loader):
        params_array, params_string = data.get_benjo_params(index=i)
        x = datapoint.flatten().to(device)
        z, mu, sigma = vae.encoder.forward(x)
        z_coords = mu.cpu().detach().numpy()
        parameter_matrix[i, :] = params_array
        latent_matrix[i, :] = z_coords
        bar()

param_data_directory = '/home/midml/Desktop/Leo_project/Benjolin_MA/params_dataset_40mfccs_beta-4_latent16.npz'
np.savez_compressed(param_data_directory, parameter_matrix=parameter_matrix, latent_matrix=latent_matrix)

print("Successfully saved parameters to ", param_data_directory)
