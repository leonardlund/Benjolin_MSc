from VAE import *
from dataloader import *
from dataloader_stats import BenjoDatasetStats
from train import *
from torch.utils.data import SubsetRandomSampler
import cupy as cp


if not torch.cuda.is_available():
    raise Exception("NO GPU AVAILABLE. ABORTING TRAINING")
device = "cuda"
torch.set_default_dtype(torch.float32)


# ---- HYPER PARAMETERS -------
data_directory = r"/home/midml/Desktop/Leo_project/Benjolin_MA/audio"
# data_directory = "/home/midml/Desktop/Leo_project/Benjolin_MA/bag-of-frames-dataset"
feature_type = 'bag-of-frames'
n_mfccs = 13
batch_size = 64
validation_split = .1
shuffle_dataset = True
random_seed = 42
beta = 0.001
learning_rate = 0.001
epochs = 5
activation = 'relu'


# ------- DATASET AND DATALOADER -----------
data = BenjoDataset(data_directory, features=feature_type, num_mfccs=n_mfccs, device=device)
data_size = data[0].shape

dataset_size = len(data)
indices = list(range(dataset_size))
split = int(cp.floor(validation_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)
train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, sampler=valid_sampler)


# ------ MODEL CREATION -------
input_dim = data_size[0] * data_size[1]
hidden_dim = input_dim // 2
latent_dim = 16

vae = VAE(input_dims=data_size, hidden_dim=hidden_dim, latent_dim=latent_dim, batch_size=batch_size)
print("model created")
vae.to(device)

# TRAINING THE MODEL
vae, train_losses, validation_losses = train(vae=vae, training_data=train_loader,
                                                     validation_data=validation_loader,
                                                     activation='relu',
                                                     epochs=epochs, opt='ADAM', beta=beta, lr=learning_rate)

# SAVING THE MODEL
save_dir = f"dir"
torch.save(vae.state_dict(), save_dir)
np.save(f"dir", train_losses)
