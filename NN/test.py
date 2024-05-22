from VAE import *
from CVAE import *
from dataloader import *
from train import *
import os
import matplotlib.pyplot as plt
from plot import plot_mfcc_spectrograms_side_by_side
import random

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: ", device)

data_directory = os.path.normpath(r"/audio")


bag_of_frames = True
feature_type = 'mfcc-bag-of-frames' if bag_of_frames else 'mfcc-2d'
data = BenjoDataset(data_directory, features=feature_type, device=device)
data_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True)

input_shape = data[0].shape
print(input_shape)
input_dim = 26 if bag_of_frames else input_shape[1] * input_shape[2]
hidden_dim = input_dim // 2
latent_dim = 2

epochs = 20

vae = VAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)  # batch_size=32
vae = vae.to(device)
vae, losses = train(vae=vae, data=data_loader, epochs=epochs)
save_dir = "/home/midml/Desktop/Leo_project/Benjolin_MA/NN/models/test-bag-1"
torch.save(vae.state_dict(), save_dir)
np.save("/home/midml/Desktop/Leo_project/Benjolin_MA/NN/models/bag-losses-1.npy", losses)
plt.plot(losses, label="Losses over epochs")

plt.rcParams['figure.dpi'] = 150

for _ in range(3):
    index = random.randint(0, 65000)
    params_array, params_string = data.get_benjo_params(index)
    example = data[index]
    if bag_of_frames:
        x_hat, z = vae.forward(example.reshape(1, 1, 26).to(device))
    else:
        x_hat, z = vae.forward(example.reshape(1, 1, input_shape[1], input_shape[2]).to(device))
    reconstructed = x_hat.cpu().detach().numpy()
    z_coords = z.cpu().detach().numpy()
    z_list = [round(z_coords[0], 3), round(z_coords[1], 3)]
    if bag_of_frames:
        plot_mfcc_spectrograms_side_by_side(example.reshape((2, 13)).T,
                                            reconstructed.reshape((2, 13)).T,
                                            benjo_params=params_string,
                                            latent_space=z_list)
    else:
        plot_mfcc_spectrograms_side_by_side(example.reshape(input_shape[1:])[:, :50],
                                            reconstructed.reshape(input_shape[1:])[:, :50],
                                            benjo_params=params_string,
                                            latent_space=z_list)

print("finished plotting")
