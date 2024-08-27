import numpy as np
from scipy.spatial import KDTree
import os

def build_kd_tree(features):
    kd_tree = KDTree(features)
    return kd_tree


def interpolate_params(kd_tree, params, x, k, snap_threshold):
    """
    latents - the latent coordinates of the dataset
    params - benjolin parameter settings of the dataset
    x - the latent space coordinate to calculate benjo params for
    snap_threshold - if current_coord is closer than snap_threshold to its closest point
                     it is considered the same point
    """
    distances, indices = kd_tree.query(x, k, workers=2)

    if k == 1:
        return params[indices[0]]
    argmin = np.argmin(distances)

    if distances[argmin] < snap_threshold:
        return params[indices[argmin], :]
    
    # here interpolation
    sum = np.sum(distances)
    proportion = distances / sum
    result = np.zeros((1, 8))
    for i in range(k):
        result = proportion[i] * params[indices[i], :]

    return result.reshape((8))

    
    

