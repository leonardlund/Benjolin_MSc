from VAE import *
from CVAE import *
from dataloader import *
from dataloader_stats import BenjoDatasetStats
from train import *
import os
import matplotlib.pyplot as plt
from plot import plot_param_reconstructions
from plot import plot_mfcc_spectrograms_side_by_side
import random
from torch.utils.data import SubsetRandomSampler


if not torch.cuda.is_available():
    raise Exception("NO GPU AVAILABLE. ABORTING TRAINING")
device = "cuda"
torch.set_default_dtype(torch.float32)


# ---- HYPER PARAMETERS -------
data_directory = os.path.normpath(r"/home/midml/Desktop/Leo_project/Benjolin_MA/audio")
# data_directory = "/home/midml/Desktop/Leo_project/Benjolin_MA/bag-of-frames-dataset"
CONTINUE_LEARNING = False
bag_of_frames = False
feature_type = 'mfcc'
n_mfccs = 40
batch_size = 32
validation_split = .1
shuffle_dataset = True
random_seed = 42
beta = 0.001
learning_rate = 0.001
gamma = 1
epochs = 5
activation = 'relu'


# ------- DATASET AND DATALOADER -----------
data = BenjoDataset(data_directory, features=feature_type, num_mfccs=n_mfccs, device=device)
data_size = data[0].shape
print(data_size)
dataset_size = len(data)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, sampler=valid_sampler)
print("dataloaders constructed")


# ------ MODEL CREATION -------
input_dim = data_size[0] * data_size[1]
print(input_dim)
# input_dim = data_size[0]
hidden_dim = input_dim // 2
latent_dim = 16

cvae = CVAE(input_dims=data_size, hidden_dim=hidden_dim, latent_dim=latent_dim, batch_size=batch_size)
print("model created")
cvae.to(device)

if CONTINUE_LEARNING:
    save_dir = "/home/midml/Desktop/Leo_project/Benjolin_MA/NN/models/param_VAE_10"
    cvae.load_state_dict(torch.load(save_dir))
    print("Loaded model from ", save_dir, " successfully!")

# TRAINING THE MODEL
vae, train_losses, validation_losses = convVAE_train(cvae=cvae, training_data=train_loader,
                                                     validation_data=validation_loader,
                                                     epochs=epochs, opt='ADAM', beta=beta, lr=learning_rate)

# SAVING THE MODEL
save_dir = f"/home/midml/Desktop/Leo_project/Benjolin_MA/NN/models/CVAE-1"
torch.save(vae.state_dict(), save_dir)
np.save(f"/home/midml/Desktop/Leo_project/Benjolin_MA/NN/models/CVAE-1-losses.npy", train_losses)
np.save(f"/home/midml/Desktop/Leo_project/Benjolin_MA/NN/models/CVAE-1-losses-val.npy", train_losses)

# PLOTTING THE RESULTS
for _ in range(3):
    index = random.randint(0, len(data))
    params_array, params_string = data.get_benjo_params(index)
    example = data[index]
    """x_hat, z = vae.forward(example.reshape(1, 1, 2*n_mfccs).to(device))
    reconstructed = x_hat.cpu().detach().numpy().reshape((2, n_mfccs)).T
    example = example.cpu().detach().numpy().reshape((2, n_mfccs)).T"""
    x_hat, z = vae.forward(example.reshape(1, -1).to(device))
    reconstructed = x_hat.view(data_size[0], -1).cpu().detach().numpy()
    example = example.view(data_size[0], -1).cpu().detach().numpy()

    z_coords = np.round(z.cpu().detach().numpy(), 3)
    # plot_param_reconstructions(example.reshape(8), reconstructed.reshape(8))
    plot_mfcc_spectrograms_side_by_side(example, reconstructed, params_array, z_coords)

plt.rcParams['figure.dpi'] = 150
plt.plot(train_losses, label="Train")
plt.plot(validation_losses, label="Validation")
plt.grid()
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss vs Epochs")
plt.show()
print("finished plotting")
