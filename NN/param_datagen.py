
from VAE import *
from dataloader import *
from train import *
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from dataloader_stats import BenjoDatasetStats


data_directory = r"dir"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: ", device)

feature_type = 'bag-of-frames'

data = BenjoDatasetStats(data_directory, features=feature_type, device=device, num_mfccs=24)
data_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)

input_shape = data[0].shape
print(input_shape)
input_dim = input_shape[0] * input_shape[1]
hidden_dim = input_dim // 2
latent_dim = 2

vae = VAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, activation='relu')  # batch_size=32

vae = vae.to(device)
model_directory = "/home/midml/Desktop/Leo_project/Benjolin_MA/NN/models/BAG-EXT-1"
vae.load_state_dict(torch.load(model_directory, map_location=torch.device))
print("Loaded model from ", model_directory, " successfully!")
plt.rcParams['figure.dpi'] = 150

parameter_matrix = cp.zeros((len(data), 8))
latent_matrix = cp.zeros((len(data), latent_dim))
bag = np.zeros((len(data), data[0].shape[0]))

for i, datapoint in enumerate(data_loader):
    params_array, params_string = data.get_benjo_params(index=i)
    x = datapoint.flatten().to(device)
    z, mu, sigma = vae.encoder.forward(x)
    z_coords = cp.asarray(mu.cpu().detach())
    parameter_matrix[i, :] = params_array
    latent_matrix[i, :] = z_coords

param_data_directory = '/home/midml/Desktop/Leo_project/Benjolin_MA/param2latent_datasets/BAG-EXT-1-latent.npz'
np.savez_compressed(param_data_directory,
                    parameter_matrix=parameter_matrix,
                    latent_matrix=latent_matrix)

print("Successfully saved parameters to ", param_data_directory)
