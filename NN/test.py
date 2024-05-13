from VAE import *
from CVAE import *
from dataloader import *
from train import *
import os
import matplotlib.pyplot as plt
from plot import plot_mfcc_spectrograms_side_by_side
import random
from torchinfo import summary


data_directory = os.path.normpath("C:/Users/Leonard/Desktop/KTH/MA Thesis/Benjolin stuff/dataset")
# print(data_directory)
data = BenjoDataset(data_directory, features='mfcc-2d', device='cpu')
data_loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)
# print(data[0])
input_shape = data[0].shape
# print(input_shape)
input_dim = input_shape[0] * input_shape[1]
hidden_dim = 32
latent_dim = 16

cvae = CVAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, batch_size=32)

cvae = train(vae=cvae, data=data_loader, epochs=5)
# summary(cvae, input_size=(32, 1, input_shape[0], input_shape[1]))

plt.rcParams['figure.dpi'] = 80

for _ in range(3):
  index = random.randint(0, 200)
  example = data[index]
  x_hat, z = cvae.forward(example.reshape(1, 1, 13, 549))
  reconstructed = x_hat.detach().numpy()
  plot_mfcc_spectrograms_side_by_side(example.reshape(13, 549), reconstructed.reshape(13, 549))
