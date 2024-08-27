import numpy as np
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt


def pre_uniformize(points):
    x_ind_sorted = np.argsort(points[:, 0])
    y_ind_sorted = np.argsort(points[:, 1])

    n = len(points)

    new_x = np.zeros((n))
    new_x[x_ind_sorted] = np.arange(n)
    new_y = np.zeros((n))
    new_y[y_ind_sorted] = np.arange(n)

    new_points = np.column_stack((new_x, new_y)) / n
    return new_points

def spread_points(points, iterations=5):
    for _ in range(iterations):
        # Create Voronoi diagram
        vor = Voronoi(points)

        # Calculate centroids of Voronoi regions
        centroids = np.array([np.mean(vor.vertices[region], axis=0)
                            for region in vor.regions
                            if len(region) > 0])
        # Update point positions to centroids
        points = centroids

    return points

if __name__ == '__main__':
    data_dir = r"C:\Users\Leonard\Desktop\benjo\latent_param_dataset_16.npz"
    # data_dir = r"C:\Users\Leonard\GitPrivate\Benjolin_MA\param2latent_datasets\CVAE-1-latent.npz"
    dataset = np.load(data_dir)
    latent = dataset['reduced_latent_matrix'][:, 0:2]
    print(latent.shape)
    x_min, x_max = np.min(latent[:, 0]), np.max(latent[:, 0])
    y_min, y_max = np.min(latent[:, 1]), np.max(latent[:, 1])
    latent[:, 0] = (latent[:, 0] - x_min) / (x_max - x_min)
    latent[:, 1] = (latent[:, 1] - y_min) / (y_max - y_min)

    points_0 = pre_uniformize(latent)
    points_1 = spread_points(points=points_0, iterations=1)
    points_2 = spread_points(points=points_1, iterations=1)
    points_3 = spread_points(points=points_2, iterations=1)
    points_4 = spread_points(points=points_3, iterations=1)

    print(points_0.shape, points_1.shape, points_2.shape, points_3.shape, points_4.shape)
    # Create a figure with 5 subplots
    fig, axes = plt.subplots(1, 5, figsize=(12, 3))

    # Plot the data on each subplot
    axes[0].scatter(points_0[:, 0], points_0[:, 1], s=0.1)
    axes[1].scatter(points_1[:, 0], points_1[:, 1], s=0.1)
    axes[2].scatter(points_2[:, 0], points_2[:, 1], s=0.1)
    axes[3].scatter(points_3[:, 0], points_3[:, 1], s=0.1)
    axes[4].scatter(points_4[:, 0], points_4[:, 1], s=0.1)

    # Set labels and titles for each subplot
    axes[0].set_title('Data 1')
    axes[1].set_title('Data 2')
    axes[2].set_title('Data 3')
    axes[3].set_title('Data 4')
    axes[4].set_title('Data 5')

    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()