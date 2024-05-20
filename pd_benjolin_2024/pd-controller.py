from pythonosc import udp_client
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
import numpy as np
from time import sleep
# import xy2params # TODO: make this function


ip = "127.0.0.1" # localhost
send_port = 8000  # must match the port declared in Pure data
listen_port = 6666
client = udp_client.SimpleUDPClient(ip, send_port) # sender
dispatcher = Dispatcher()
server = BlockingOSCUDPServer((ip, listen_port), dispatcher) # listener

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
    xy = np.array([int(xy[0]), int(xy[1])])
    # params = xy2params.xy2params(xy_list)
    params = np.random.randint(0, 126, size=(8))
    params_message = '-'.join([str(param) for param in params])
    client.send_message("/params", params_message)
    print("xy received... params sent!")
    


dispatcher.map("/print", print_handler)
dispatcher.map("/xy", xy_handler)
dispatcher.map("/quit", shutdown_handler)
dispatcher.set_default_handler(default_handler)
server.serve_forever()  # Blocks forever
