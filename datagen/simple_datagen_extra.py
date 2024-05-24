import itertools
from copy import copy
import os
from time import sleep
from pythonosc import udp_client
import subprocess
import numpy as np

global pd_executable
# find the pd executable in your computer, the following works on mac

MACHINE = "mid-ml"  # "laptop"

if MACHINE == "mid-ml":
    pd_executable = r'/usr/lib/puredata/bin/pd'
    pd_script_path = r'/home/midml/Desktop/Leo_project/Benjolin_MA/pd_benjolin_2024/pd-benjo-leo.pd'
else:
    pd_executable = r'C:\"Program Files\Pd\bin\pd"'
    pd_script_path = os.path.normpath(r"C:\Users\Leonard\GitPrivate\Benjolin_MA\pd_benjolin_2024\pd-benjo-leo.pd")

port = 5005  # must match the port declared in Pure data
client = udp_client.SimpleUDPClient("127.0.0.1", port)

knob_resolution = 126
num_settings_per_knob = 4
setting_list = []


num_combinations = 5000
progress = 0
progress_bar = ""
antiprogress_bar = "-" * 100
for i in range(num_combinations):
    combo = ''
    parameters = []
    for k in range(8):
        parameters.append(np.random.randint(1, knob_resolution))
        if k == 2 and parameters[2] == parameters[0]:
            parameters[2] += 1
        combo += f';param{k+1} {str(parameters[k])} '
    combo += ';'

    # command = pd_executable + f' -nogui -send "{combo}"  ' + pd_script_path
    command = pd_executable + f' -nogui -noverbose -send "{combo}"  ' + pd_script_path + ' 1> /dev/null'
    
    os.system(command)

    new_progress = int(100 * (i+1) / num_combinations)
    if new_progress > progress:
        progress_bar += "#"
        progress = new_progress
        antiprogress_bar = antiprogress_bar[:-1]
    bar = progress_bar + antiprogress_bar
    print(f"{bar}.    Sound number {i+1} of {num_combinations}. Progress: {round(100 * (i+1) / num_combinations, 2)}%")

print("Program finished execution")
