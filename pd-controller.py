from pythonosc import udp_client
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
import numpy as np
from NN.latent2param_MLP import MLPRegressor
from NN.VAE import VAE
import torch
from time import sleep


def get_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Example usage
    # model = MLPRegressor(input_dim=2, hidden_dim=4, output_dim=8)
    model = VAE(input_dim=8, hidden_dim=16, latent_dim=2, activation='tanh', device=device)
    model = model.to(device).float()
    # model_dir = "/home/midml/Desktop/Leo_project/Benjolin_MA/NN/models/mlp_regressor-1"
    model_dir = "/home/midml/Desktop/Leo_project/Benjolin_MA/NN/models/param_VAE_6"
    model.load_state_dict(torch.load(model_dir))
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
    params = MODEL.decoder.forward(z=xy)
    params = list(params.cpu().detach().numpy() * 127)
    print(params)
    params_message = '-'.join([str(int(param)) for param in params])
    client.send_message("/params", params_message)
    # print("xy received... params sent!")


if __name__ == "__main__":
    MODEL, DEVICE = get_model()
    ip = "127.0.0.1"  # localhost
    send_port = 8000  # must match the port declared in Pure data
    listen_port = 6666
    client = udp_client.SimpleUDPClient(ip, send_port)  # sender
    dispatcher = Dispatcher()
    server = BlockingOSCUDPServer((ip, listen_port), dispatcher)  # listener

    dispatcher.map("/print", print_handler)
    dispatcher.map("/xy", xy_handler)
    dispatcher.map("/quit", shutdown_handler)
    dispatcher.set_default_handler(default_handler)
    print("Set up complete! Start playing the benjolin!")
    server.serve_forever()  # Blocks forever
