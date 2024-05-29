import matplotlib.pyplot as plt
from VAE import *
from dataloader import *
import numpy as np
import os
import random
import torch


def plot_mfcc_spectrograms_side_by_side(mfccs1, mfccs2, benjo_params, latent_space):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # Create a figure with 2 subplots

    for i, (mfccs, ax) in enumerate(zip([mfccs1.T, mfccs2.T], axes)):
        ax.imshow(mfccs.T, cmap='inferno', aspect='auto', origin='lower')
        ax.set_title(f'MFCC Spectrogram {"Original" if i==0 else "Reconstructed"}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Mel Frequency')
    plt.suptitle(f'Original and Reconstructed MFCCs for Benjolin parameters: {benjo_params}' +
                 f'\nLocation in latent space: {latent_space}')
    plt.tight_layout()
    plt.show()


def plot_param_reconstructions(param_real, param_reconstructed):
    param_real = np.round(param_real, 2)
    param_reconstructed = np.round(param_reconstructed, 2)
    x = np.arange(param_real.shape[0]) + 1
    width = 0.25
    fig, axes = plt.subplots(layout='constrained')

    rects = axes.bar(x - width/2, param_real, width, label='Real')
    axes.bar_label(rects, padding=3)
    rects = axes.bar(x + width/2, param_reconstructed, width, label='Reconstructed')
    axes.bar_label(rects, padding=3)
    axes.set_ylabel('Setting')
    axes.set_xlabel('Parameter')
    axes.set_xticks(x)
    axes.set_title('Reconstructions of benjolin parameters')
    axes.legend()
    plt.show()


def plot_latent(latent_data, params):
    param_names = ['01_FRQ', '01_RUN', '02_FRQ', '02_RUN', 'FIL_FRQ', 'FIL_RES', 'FIL_RUN', 'FIL_SWP']
    cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
    fig, axes = plt.subplots(2, 4, figsize=(15, 10))
    for i in range(2):
        for j in range(4):
            axes[i, j].scatter(latent_data[:, 0], latent_data[:, 1],
                               c=params[:, i], marker='o', alpha=0.5, s=0.2, cmap=cmaps[j])
            axes[i, j].set_title(f'Latent space\ncolored according to {param_names[i*4 + j]}')
            axes[i, j].set_xlabel('First latent dimension')
            axes[i, j].set_ylabel('Second latent dimension')
    plt.suptitle("Latent space")
    plt.tight_layout()
    plt.show()


def plot_correlation(latent_data, params, knob='01_FRQ'):
    param_names = ['01_FRQ', '01_RUN', '02_FRQ', '02_RUN', 'FIL_FRQ', 'FIL_RES', 'FIL_RUN', 'FIL_SWP']
    param_index = param_names.index(knob)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for j in range(2):
        axes[j].scatter(params[:, param_index], latent_data[:, j], marker='o', alpha=0.5, s=0.2)
        axes[j].set_title(f'Correlation between {knob} and {j}th latent dimension')
        axes[j].set_ylabel(f'{j}th latent dimension')
        axes[j].set_xlabel(f'{knob} setting')
    plt.suptitle(f"Correlation between latent dims and {knob}")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    data_directory = os.path.normpath(r"/home/midml/Desktop/Leo_project/Benjolin_MA/audio")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: ", device)

    bag_of_frames = True
    # feature_type = 'mfcc-bag-of-frames' if bag_of_frames else 'mfcc-2d'
    feature_type = 'params'
    n_mfccs = 4
    data = BenjoDataset(data_directory, features=feature_type, device=device, num_mfccs=n_mfccs)
    data_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True)

    input_shape = data[0].shape
    print(input_shape)
    input_dim = 2 * n_mfccs if bag_of_frames else input_shape[1] * input_shape[2]
    hidden_dim = 16
    latent_dim = 2

    vae = VAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)  # batch_size=32

    vae = vae.to(device)
    save_dir = "/home/midml/Desktop/Leo_project/Benjolin_MA/NN/models/param_VAE_4"
    vae.load_state_dict(torch.load(save_dir))
    print("Loaded model from ", save_dir, " successfully!")
    plt.rcParams['figure.dpi'] = 150

    for _ in range(3):
        index = random.randint(65000, 70000)
        params_array, params_string = data.get_benjo_params(index)
        example = data[index]
        if bag_of_frames:
            z, mu, log_var = vae.encoder.forward(example.reshape(1, 1, 2 * n_mfccs).to(device))
            x_hat = vae.decoder.forward(mu)
            # x_hat, z = vae.forward(example.reshape(1, 1, 2 * n_mfccs).to(device))
            reconstructed = x_hat.cpu().detach().numpy().reshape((2, n_mfccs)).T
            example = example.cpu().detach().numpy().reshape((2, n_mfccs)).T
        else:
            x_hat, z = vae.forward(example.reshape(1, 1, input_shape[1], input_shape[2]).to(device))
            reconstructed = x_hat.cpu().detach().numpy().reshape(input_shape[1:])
            example = example.cpu().detach().numpy().reshape(input_shape[1:])

        z_coords = np.round(z.cpu().detach().numpy(), 3)
        # plot_mfcc_spectrograms_side_by_side(example, reconstructed, benjo_params=params_string, latent_space=z_coords)
        plot_param_reconstructions(example.reshape(8), reconstructed.reshape(8))

    print("finished plotting")
