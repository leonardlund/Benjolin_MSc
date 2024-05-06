from VAE import *
from dataloader import *
from train import *
import os
import matplotlib.pyplot as plt
from plot import plot_mfcc_spectrograms_side_by_side
import random


data_directory = os.path.normpath("C:/Users/Leonard/Desktop/KTH/MA Thesis/Benjolin stuff/dataset")
# print(data_directory)
data = BenjoDataset(data_directory, features='mfcc-2d', device='cpu')

print(data[0])
input_shape = data[0].shape
print(input_shape)
input_dim = input_shape[0] * input_shape[1]
hidden_dim = input_dim // 2
latent_dim = 2

vae = VAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)

vae = train(vae=vae, data=data, epochs=3)

plt.rcParams['figure.dpi'] = 80

for _ in range(3):
  index = random.randint(0, 100)
  example = data[index]
  x_hat, z = vae.forward(data[index].flatten())
  reconstructed = x_hat.detach().numpy()
  plot_mfcc_spectrograms_side_by_side(example, reconstructed)
