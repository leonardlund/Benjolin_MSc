from VAE import *
from dataloader import *
from train import *
import os
import matplotlib.pyplot as plt
from plot import plot_param_reconstructions
import random
from torch.utils.data import SubsetRandomSampler


if not torch.cuda.is_available():
    raise Exception("NO GPU AVAILABLE. ABORTING TRAINING")
device = "cuda"

torch.set_default_dtype(torch.float32)
data_directory = os.path.normpath(r"/home/midml/Desktop/Leo_project/Benjolin_MA/audio")
CONTINUE_LEARNING = False
bag_of_frames = True
feature_type = 'params'
n_mfccs = 4
batch_size = 32
validation_split = .1
shuffle_dataset = True
random_seed = 42
input_dim, hidden_dim, latent_dim = 8, 16, 2
beta = 0.001
learning_rate = 0.002
gamma = 1
epochs = 15
activation = 'tanh'

data = BenjoDataset(data_directory, features=feature_type, device=device)

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

vae = VAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, activation=activation)  # batch_size=32
vae.to(device)

if CONTINUE_LEARNING:
    save_dir = "/home/midml/Desktop/Leo_project/Benjolin_MA/NN/models/param_VAE_9"
    vae.load_state_dict(torch.load(save_dir))
    print("Loaded model from ", save_dir, " successfully!")

vae, train_losses, validation_losses = train(vae=vae, training_data=train_loader, validation_data=validation_loader,
                                             epochs=epochs, opt='ADAM', beta=beta, lr=learning_rate, gamma=gamma)

save_dir = f"/home/midml/Desktop/Leo_project/Benjolin_MA/NN/models/param_VAE_9"
torch.save(vae.state_dict(), save_dir)
np.save(f"/home/midml/Desktop/Leo_project/Benjolin_MA/NN/models/param_VAE_9_train_losses.npy", train_losses)
np.save(f"/home/midml/Desktop/Leo_project/Benjolin_MA/NN/models/param_VAE_9_val_losses.npy", train_losses)


for _ in range(3):
    index = random.randint(0, len(data))
    params_array, params_string = data.get_benjo_params(index)
    example = data[index]
    x_hat, z = vae.forward(example.reshape(1, 1, 2*n_mfccs).to(device))
    reconstructed = x_hat.cpu().detach().numpy().reshape((2, n_mfccs)).T
    example = example.cpu().detach().numpy().reshape((2, n_mfccs)).T
    """else:
        x_hat, z = vae.forward(example.reshape(1, 1, input_shape[1], input_shape[2]).to(device))
        reconstructed = x_hat.cpu().detach().numpy().reshape(input_shape[1:])
        example = example.cpu().detach().numpy().reshape(input_shape[1:])"""

    z_coords = np.round(z.cpu().detach().numpy(), 3)
    plot_param_reconstructions(example.reshape(8), reconstructed.reshape(8))

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
