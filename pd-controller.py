from pythonosc import udp_client
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
import numpy as np
from NN.latent2param_MLP import MLPRegressor
from NN.VAE import VAE
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl


def get_model(model_path, architecture='vae', device='cuda'):
    if architecture == 'vae':
        model = VAE(input_dim=8, hidden_dim=16, latent_dim=2, activation='tanh', device=device)
    else:  # architecture == 'mlp'
        model = MLPRegressor(input_dim=2, hidden_dim=16, output_dim=8)
    model = model.to(device).float()
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    return model, device


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
    xy = torch.tensor(np.array([int(xy[0]), int(xy[1])])).to(DEVICE).float()
    old_min, old_max = 0, 127
    new_min, new_max = -1.5, 1.1
    xy[0] = ((xy[0] - old_min) / (old_max - old_min)) / (new_max - new_min) + new_min
    new_min, new_max = -1, 1
    xy[1] = ((xy[1] - old_min) / (old_max - old_min)) / (new_max - new_min) + new_min

    if ARCHITECTURE == 'vae':
        params = MODEL.decoder.forward(z=xy) * 127
    else:
        params = MODEL(xy) * 127
    params = list(params.cpu().detach().numpy())

    print(params)
    params_message = '-'.join([str(int(param)) for param in params])
    client.send_message("/params", params_message)


if __name__ == "__main__":
    # model_path = "/home/midml/Desktop/Leo_project/Benjolin_MA/NN/models/mlp-bag3"
    # model_path = r'C:\Users\Leonard\GitPrivate\Benjolin_MA\NN\models\BAG-EXT-1'
    model_path = r'C:\Users\Leonard\GitPrivate\Benjolin_MA\NN\models\mlp-bag3'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ARCHITECTURE = 'mlp'
    MODEL, DEVICE = get_model(model_path, architecture=ARCHITECTURE, device=device)
    ip = "127.0.0.1"  # localhost
    send_port = 8000  # must match the port declared in Pure data
    listen_port = 6666
    client = udp_client.SimpleUDPClient(ip, send_port)  # sender
    dispatcher = Dispatcher()

    mpl.use('Qt5Agg')

    # data_dir = '/home/midml/Desktop/Leo_project/Benjolin_MA/param2latent_datasets/BAG-EXT-1-latent.npz'
    data_dir = r'C:\Users\Leonard\GitPrivate\Benjolin_MA\param2latent_datasets\BAG-EXT-1-latent.npz'
    dataset = np.load(data_dir)
    latent = dataset['latent_matrix']
    parameter = dataset['parameter_matrix']
    bag = dataset["mfccs"]


    def on_click(event):
        index = event.ind[0]
        print(index)
        params = parameter[index]
        print(params)
        # plt.imshow(bag[index, :, :].reshape((12, 12)), cmap="inferno")
        # plt.draw()
        params_message = '-'.join([str(int(param)) for param in params])
        client.send_message("/params", params_message)


    fig, ax = plt.subplots()
    ax.scatter(latent[:, 0], latent[:, 1], alpha=0.8, s=0.3, picker=True)
    connection_id = fig.canvas.mpl_connect('pick_event', on_click)
    plt.isinteractive()
    plt.show()

    server = BlockingOSCUDPServer((ip, listen_port), dispatcher)  # listener

    dispatcher.map("/print", print_handler)
    dispatcher.map("/xy", xy_handler)
    dispatcher.map("/quit", shutdown_handler)
    dispatcher.set_default_handler(default_handler)
    print("Set up complete! Start playing the benjolin!")
    server.serve_forever()  # Blocks forever
