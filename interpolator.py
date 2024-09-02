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


def find_next_point(kd_tree, latents, params, x, index_x, index_y, y, k=10, snap_threshold=0.001):
    """
    given two points, returns a list of points that describes the path of least 
    resistance (in parameter space) between the two points
    """
    param_distance_0 = np.linalg.norm(params[index_x] - params[index_y])
    latent_distance_0 = np.linalg.norm(x-y)

    param_space_distances = np.array([])
    latent_space_distances = np.array([])

    distances, indices = kd_tree.query(x, k, workers=2)

    for i in range(k):
        param_distance = np.linalg.norm(params[indices[k]] - params[index_x])
        latent_distance = np.linalg.norm(y - x)
        param_space_distances.append(param_distance)
        latent_space_distances.append(latent_distance)
    
    argmin = np.argmin(param_space_distances)

    if latent_space_distances[argmin] < latent_distance_0:
        return latents[indices[argmin]]
    else:
        raise Exception
    

        
def create_path_between_2_points(kd_tree, latents, params, x, index_x, index_y, y, k=5, snap_threshold=0.001):
    """
    given two points, returns a list of points that describes the path of least 
    resistance (in parameter space) between the two points
    """
    param_distance_0 = np.linalg.norm(params[index_x] - params[index_y])
    latent_distance_0 = np.linalg.norm(x-y)

    param_space_distances = []
    latent_space_distances = []

    distances, indices = kd_tree.query(x, k, workers=2)

    for i in range(k):
        param_distance = np.linalg.norm(params[indices[k]] - params[index_x])
        latent_distance = np.linalg.norm(y - x)
        param_space_distances.append(param_distance)
        latent_space_distances.append(latent_distance)

    
    
        

