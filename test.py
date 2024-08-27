import numpy as np
import interpolator
import timeit


setup = '''
import interpolator
import os
import numpy as np
data_dir = os.path.normpath('C:/Users/Leonard/Desktop/benjo/latent_param_dataset_16.npz')
dataset = np.load(data_dir)
features = dataset['reduced_latent_matrix']
features = features[:, :2]
parameter = dataset['parameter_matrix']
kd_tree = interpolator.build_kd_tree(features=features)
'''

stmt = '''
x = np.random.normal(size=(2))
new_params = interpolator.interpolate_params(kd_tree, features, parameter, x=x, k=3, snap_threshold=20)
'''

print(timeit.timeit(setup=setup, stmt=stmt, number=10000))