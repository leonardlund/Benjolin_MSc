from VAE import *
from CVAE import *
from dataloader import *
from train import *
import os
import matplotlib.pyplot as plt
from plot import plot_mfcc_spectrograms_side_by_side
import random
# from tsnecuda import TSNE

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: ", device)

data_directory = os.path.normpath(r"/home/midml/Desktop/Leo_project/Benjolin_MA/audio")


bag_of_frames = True
feature_type = 'mfcc-bag-of-frames' if bag_of_frames else 'mfcc-2d'
n_mfccs = 40
data = BenjoDataset(data_directory, features=feature_type, device=device, num_mfccs=n_mfccs, )
data_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True)

input_shape = data[0].shape
print(input_shape)
input_dim = n_mfccs * 2 if bag_of_frames else input_shape[1] * input_shape[2]
hidden_dim = input_dim // 2
latent_dim = 16

epochs = 20

for k in range(2):
    vae = VAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)  # batch_size=32
    vae = vae.to(device)
    if k == 0:
        beta = 1e-4
    else:
        beta = 1e-5
    vae, losses = train(vae=vae, data=data_loader, epochs=epochs, beta=beta)
    save_dir = f"/home/midml/Desktop/Leo_project/Benjolin_MA/NN/models/test-bag-beta1e-{4+k}-40mfcc"
    torch.save(vae.state_dict(), save_dir)
    np.save(f"/home/midml/Desktop/Leo_project/Benjolin_MA/NN/models/bag-losses-beta1e-{4+k}-40mfcc.npy", losses)

    for _ in range(3):
        index = random.randint(0, 65000)
        params_array, params_string = data.get_benjo_params(index)
        example = data[index]
        if bag_of_frames:
            x_hat, z = vae.forward(example.reshape(1, 1, 180).to(device))
        else:
            x_hat, z = vae.forward(example.reshape(1, 1, input_shape[1], input_shape[2]).to(device))
        reconstructed = x_hat.cpu().detach().numpy()
        example = example.cpu().detach().numpy()
        z_coords = z.cpu().detach().numpy()
        z_list = [round(z_coords[0], 3), round(z_coords[1], 3)]
        if bag_of_frames:
            plot_mfcc_spectrograms_side_by_side(example.reshape((2, 90)).T,
                                                reconstructed.reshape((2, 90)).T,
                                                benjo_params=params_string,
                                                latent_space=z_list)
        else:
            plot_mfcc_spectrograms_side_by_side(example.reshape(input_shape[1:])[:, :50],
                                                reconstructed.reshape(input_shape[1:])[:, :50],
                                                benjo_params=params_string,
                                                latent_space=z_list)

    plt.rcParams['figure.dpi'] = 150
    plt.plot(losses, label="Losses over epochs")
    plt.grid()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()
    print("finished plotting")
