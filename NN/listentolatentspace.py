import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use('Qt5Agg')

data_dir = '/home/midml/Desktop/Leo_project/Benjolin_MA/param2latent_datasets/bag-vae-3-latent.npz'
dataset = np.load(data_dir)
latent = dataset['latent_matrix']
parameter = dataset['parameter_matrix']


def on_click(event):
    index = event.ind[0]
    print(index)
    sample = parameter[index]


fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', sharey='col')
ax1.scatter(latent[:, 0], latent[:, 1], alpha=0.8, s=0.3, picker=True)
connection_id = fig.canvas.mpl_connect('pick_event', on_click)
plt.isinteractive()
plt.show()

