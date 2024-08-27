from pythonosc import udp_client
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
import numpy as np
from scipy.spatial import KDTree


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


def print_handler(address, *args):
    print(f"{address}: {args}")


def default_handler(address, *args):
    print(f"DEFAULT {address}: {args}")


def shutdown_handler(address, *args):
    print("shutdown")
    server.shutdown()
    print("Shutting down...")


def xy_handler(address, xy_symbol):
    xy = str(xy_symbol)
    xy = xy.split('-')[:2]
    xy = [float(xy[0]), float(xy[1])]

    setting = interpolate_params(kd_tree=kd_tree,
                                 params=parameter, 
                                 x=xy, k=2,
                                 snap_threshold=0.1)

    params_message = '-'.join([str(int(param)) for param in setting])
    # print(xy, params_message)
    client.send_message("/params", params_message)


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


if __name__ == "__main__":
    ip = "127.0.0.1"  # localhost
    send_port = 8000  # must match the port declared in Pure data
    listen_port = 6666
    client = udp_client.SimpleUDPClient(ip, send_port)  # sender
    dispatcher = Dispatcher()

    # data_dir = r"latent_dataset.npz"
    # data_dir = r"C:\Users\Leonard\GitPrivate\Benjolin_MA\param2latent_datasets\CVAE-1-latent.npz"
    data_dir = r"C:\Users\Leonard\Desktop\benjo\latent_param_dataset_16.npz"
    dataset = np.load(data_dir)
    latent = dataset['reduced_latent_matrix']

    latent = pre_uniformize(latent)

    parameter = dataset['parameter_matrix']
    kd_tree = build_kd_tree(features=latent[:, :2])

    server = BlockingOSCUDPServer((ip, listen_port), dispatcher)  # listener

    dispatcher.map("/print", print_handler)
    dispatcher.map("/xy", xy_handler)
    dispatcher.map("/quit", shutdown_handler)
    dispatcher.set_default_handler(default_handler)
    print("Set up complete! Start playing the benjolin!")
    server.serve_forever()  # Blocks forever

