import itertools
from copy import copy
import os
from time import sleep
from pythonosc import udp_client
import subprocess
# from pyuac import main_requires_admin
import pyuac

global pd_executable
# find the pd executable in your computer, the following works on mac
pd_executable = r'C:\"Program Files\Pd\bin\pd"'
pd_script_path = os.path.normpath(r"C:\Users\Leonard\GitPrivate\Benjolin_MA\pd_benjolin_2024\pain.pd")

port = 5005  # must match the port declared in Pure data
client = udp_client.SimpleUDPClient("127.0.0.1", port)

knob_resolution = 126
num_settings_per_knob = 2
setting_list = []
for i in range(num_settings_per_knob):
    setting = int(knob_resolution * (i+1)/(num_settings_per_knob+1))
    setting_list.append(setting)
parameters = [
    ("param1", copy(setting_list)),
    ("param2", copy(setting_list)),
    ("param3", copy(setting_list)),
    ("param4", copy(setting_list)),
    ("param5", copy(setting_list)),
    ("param6", copy(setting_list)),
    ("param7", copy(setting_list)),
    ("param8", copy(setting_list))
]

# Generate all combinations of parameter settings
combinations = list(itertools.product(*[param[1] for param in parameters]))

for j, combination in enumerate(combinations):
    if j > 0 and False:
        break
    combo = ''
    for i, param in enumerate(combination):
        if i == 0:
            param += 1
        combo += f';param{i+1} {str(param)} '
    combo += ';'

    command = pd_executable + f' -nogui -send "{combo}"  ' + pd_script_path
    print("Sent bash command: ", command)
    
    os.system(command)
    # client.send_message("/startup", osc_string)
    # print("Sent OSC: ", osc_string)

    # include -nogui flag to execute pd without GUI
    #command = pd_executable + f' -send "; filename {filename}" -nogui ' + pd_script_path

