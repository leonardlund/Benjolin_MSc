import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

data_dir = ('/home/midml/Desktop/Leo_project/Benjolin_MA/'
            'param2latent_datasets/params_dataset_40mfccs_beta-4_latent16.npz')
dataset = np.load(data_dir)
latent = dataset['latent_matrix']
parameter = dataset['parameter_matrix']

tsne = True
if tsne:
    parameter_tsne = TSNE(n_components=2).fit_transform(parameter)


def plot_latent(latent_data, params):
    param_names = ['01_FRQ', '01_RUN', '02_FRQ', '02_RUN', 'FIL_FRQ', 'FIL_RES', 'FIL_RUN', 'FIL_SWP']
    cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
    fig, axes = plt.subplots(2, 4, figsize=(15, 10))
    for i in range(2):
        for j in range(4):
            axes[i, j].scatter(latent_data[:, 0], latent_data[:, 1],
                               c=params[:, i], marker='o', alpha=0.5, s=0.2, cmap=cmaps[j])
            axes[i, j].set_title(f'Latent space\ncolored according to the {param_names[i*4 + j]}')
            axes[i, j].set_xlabel('First latent dimension')
            axes[i, j].set_ylabel('Second latent dimension')
    plt.suptitle("Latent space")
    plt.tight_layout()
    plt.show()


def plot_correlation(latent_data, params, knob='01_FRQ'):
    param_names = ['01_FRQ', '01_RUN', '02_FRQ', '02_RUN', 'FIL_FRQ', 'FIL_RES', 'FIL_RUN', 'FIL_SWP']
    param_index = param_names.index(knob)

    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    for i in range(4):
        for j in range(4):
            axes[i, j].scatter(params[:, param_index], latent_data[:, i * 4 + j], marker='o', alpha=0.5, s=0.2)
            axes[i, j].set_title(f'Correlation between {knob} and {i * 4 + j}th latent dimension')
            axes[i, j].set_ylabel(f'{i*4+j}th latent dimension')
            axes[i, j].set_xlabel(f'{knob} setting')
    plt.suptitle(f"Correlation between latent dims and {knob}")
    plt.tight_layout()
    plt.show()


# plot_correlation(latent, parameter, '01_RUN')
plt.scatter(parameter_tsne[0], parameter_tsne[1], alpha=0.5, s=0.2)
