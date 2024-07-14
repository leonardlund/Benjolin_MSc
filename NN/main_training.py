from VAE import *
from dataloader import BenjoDataset
from train import *
from torch.utils.data import SubsetRandomSampler
import cupy as cp
import numpy as np
import pickle
import argparse
import configparser
import sys
import os


# ---- HYPER PARAMETERS -------
"""
- This is where you change things about the model and the dataset.
- For a feature to be excluded in the bag-of-frames feature tensor change the corresponding boolean to False
 (and vice versa)
"""
# NOTE this code is stolen from Kivanc Tatar
#Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default ='./default.ini' , help='path to the config file')
args = parser.parse_args()

#Get configs
config_path = args.config
config = configparser.ConfigParser(allow_no_value=True)
try: 
  config.read(config_path)
except FileNotFoundError:
  print('Config File Not Found at {}'.format(config_path))
  sys.exit()


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
weight_normalization = True
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


output_directory = "cephyr/users/lundle/Alvis/benjo/runs/"

# NOTE this code is also stolen from Kivanc
if not continue_training:
    run_id = run_number
    while True:
        try:
            os.makedirs(output_directory + f'/run_{run_number}')
            break
        except OSError:
            if output_directory.is_dir():
                run_id = run_id + 1
                continue
            raise 'Broken directory'
else:
    raise 'TODO: handle this'

print("Workspace: {}".format(output_directory))


# ------- DATASET AND DATALOADER -----------
# stat_dict = pickle.load(open(stat_dict_path, "rb"))
stat_dict = None
data = BenjoDataset(data_path, features=feature_type, num_mfccs=n_mfccs, device=device,
                    fft_args=window_args, weight_normalization=weight_normalization,
                    feature_dict=feature_dict, stat_dictionary=stat_dict)
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
input_dim = data_size[0] * data_size[1]
hidden_dim = input_dim // 2

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
