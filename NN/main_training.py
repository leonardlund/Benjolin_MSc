from VAE import *
from dataloader_librosa import BenjoDataset
from train import *
from torch.utils.data import SubsetRandomSampler
import cupy as cp
import numpy as np
import pickle
import argparse
import configparser
import sys
import os
from pathlib import Path
from sklearn.decomposition import PCA


# ---- HYPER PARAMETERS -------
"""
- This is where you change things about the model and the dataset.
- For a feature to be excluded in the bag-of-frames feature tensor change the corresponding boolean to False
 (and vice versa)
"""
# NOTE this code is stolen from Kivanc Tatar
#Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default ='/cephyr/users/lundle/Alvis/benjo/Benjolin_MA/default.ini' , help='path to the config file')
args = parser.parse_args()

#Get configs
config_path = args.config
print("Config path: ", config_path)
config = configparser.ConfigParser(allow_no_value=True)
try: 
  config.read(config_path)
except FileNotFoundError:
  print('Config File Not Found at {}'.format(config_path))
  sys.exit()
print(config.sections())

sample_rate = config['audio'].getint('sample_rate')

data_path = config['dataset'].get('data_path')
model_path = config['dataset'].get('model_path')

latent_dim = config['vae'].getint('latent_dim')
kl_beta = config['vae'].getfloat('kl_beta')
output_activation = config['vae'].get('output_activation')

epochs = config['training'].getint('epochs')
learning_rate = config['training'].getfloat('learning_rate')
batch_size = config['training'].getint('batch_size')
continue_training = config['training'].getboolean('continue_training')
patience = config['training'].getint('patience')
run_number = config['training'].getint('run_number')

feature_type = config['features'].get('feature_type')
hop_length = config['features'].getint('hop_length')
window_length = config['features'].getint('window_length')
pad = config['features'].getint('pad')
n_mfccs = config['features'].getint('n_mfccs')
mfcc = config['features'].getboolean('mfcc')
centroid = config['features'].getboolean('centroid')
zcr = config['features'].getboolean('zcr')
rms = config['features'].getboolean('rms')
flux = config['features'].getboolean('flux')
flatness = config['features'].getboolean('flatness')
weight_normalization = config['features'].getboolean('weight_normalization')
window_args = {'win_length': window_length, 'hop_length':hop_length, 'pad': pad}
feature_dict = {'mfcc': mfcc, 'rms': rms, 'zcr':zcr, 'centroid':centroid, 'flux':flux, 'flatness':flatness}



# data_directory = r"/home/midml/Desktop/Leo_project/Benjolin_MA/audio"
# stat_dict_path = r"dir"
# data_directory = "/home/midml/Desktop/Leo_project/Benjolin_MA/bag-of-frames-dataset"
validation_split = .1
shuffle_dataset = True
random_seed = 42



# ------- Torch settings --------------
if not torch.cuda.is_available():
    raise Exception("NO GPU AVAILABLE. ABORTING TRAINING")
device = "cuda"
torch.set_default_dtype(torch.float32)


output_directory = "/cephyr/users/lundle/Alvis/benjo/runs/"

# NOTE this code is also stolen from Kivanc
if not continue_training:
    run_id = run_number
    while True:
        try:
            working_directory = output_directory + f'/run_{run_id}'
            os.makedirs(working_directory)
            os.system(f'cp /cephyr/users/lundle/Alvis/benjo/Benjolin_MA/default.ini {working_directory + "/settings.ini"}')
            break
        except OSError:
            if Path(output_directory).is_dir():
                run_id = run_id + 1
                continue
            raise 'Broken directory'
else:
    raise 'TODO: handle this'

# print("Workspace: {}".format(output_directory))


# ------- DATASET AND DATALOADER -----------
# stat_dict = pickle.load(open(stat_dict_path, "rb"))
stat_dict = None
print(data_path)
"""
data = BenjoDataset(data_path, features=feature_type, num_mfccs=n_mfccs, device=device,
                    fft_args=window_args, weight_normalization=weight_normalization,
                    feature_dict=feature_dict, stat_dictionary=stat_dict)
"""
data = BenjoDataset(data_path, mfcc_normalization=weight_normalization)
data_size = data[0].shape
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


# ------ MODEL CREATION -------
input_dim = data_size[0]
hidden_dim = input_dim // 2

vae = VAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
print("model created")
vae.to(device)

# TRAINING THE MODEL
vae, train_losses, validation_losses = train(vae=vae, training_data=train_loader,
                                                     validation_data=validation_loader,
                                                     epochs=epochs, opt='ADAM', beta=kl_beta, lr=learning_rate)

# SAVING THE MODEL
model_save_dir = working_directory + '/model'
train_losses_save_dir = working_directory + '/training_losses'
val_losses_save_dir = working_directory + '/validation_losses'
torch.save(vae.state_dict(), model_save_dir)
np.save(train_losses_save_dir, train_losses)
np.save(val_losses_save_dir, validation_losses)


data_loader = torch.utils.data.DataLoader(data, batch_size=1)


parameter_matrix = np.zeros((len(data), 8))
latent_matrix = np.zeros((len(data), latent_dim))
sigma_matrix = np.zeros((len(data), latent_dim))

for i, datapoint in enumerate(data_loader):
    params_array, params_string = data.get_benjo_params(index=i)
    x = datapoint.flatten().to(device)
    z, mu, sigma = vae.encoder.forward(x)
    parameter_matrix[i, :] = params_array
    latent_matrix[i, :] = mu.cpu().detach().numpy()
    sigma_matrix[i, :] = sigma.cpu().detach().numpy()


param_data_directory = working_directory + f'/latent_param_dataset_{run_id}.npy'

if latent_dim >= 3:
    pca_model = PCA(n_components=3)
    pca_latent = pca_model.fit_transform(latent_matrix)


    np.savez_compressed(param_data_directory,
                        parameter_matrix=parameter_matrix,
                        latent_matrix=latent_matrix,
                        sigma_matrix=sigma_matrix,
                        reduced_latent_matrix=pca_latent)
else:
    np.savez_compressed(param_data_directory,
                        parameter_matrix=parameter_matrix,
                        sigma_matrix=sigma_matrix,
                        latent_matrix=latent_matrix)

print("Successfully saved latent-param datset to ", param_data_directory)
