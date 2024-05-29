import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

data_dir = '/home/midml/Desktop/Leo_project/Benjolin_MA/audio/'
files = [f.removesuffix('.wav').split('-') for f in os.listdir(data_dir) if f.endswith('.wav')]
for file in files:
    file = [int(p) for p in file]
params_array = np.array(files, dtype='int')
print(params_array.shape)
print(params_array.dtype)

parameter_tsne = TSNE(n_components=2).fit_transform(params_array[:4000, :])
print(parameter_tsne.shape)

plt.scatter(parameter_tsne[:, 0], parameter_tsne[:, 1], c=params_array[:4000, 0], alpha=0.5, s=0.75)
plt.show()
