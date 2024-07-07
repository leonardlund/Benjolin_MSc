import itertools
from copy import copy
import os
from time import sleep
from pythonosc import udp_client
import subprocess
# from pyuac import main_requires_admin



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

num_combinations = len(combinations)
progress = 0
progress_bar = ""
antiprogress_bar = "-" * 100
for j, combination in enumerate(combinations):
    combo = ''
    for i, param in enumerate(combination):
        if i == 0:
            param += 1
        combo += f';param{i+1} {str(param)} '
    combo += ';'

    # command = pd_executable + f' -nogui -send "{combo}"  ' + pd_script_path
    command = pd_executable + f' -nogui -noverbose -send "{combo}"  ' + pd_script_path + ' 1> /dev/null'
    # print("Sending bash command: ", command)
    
    os.system(command)

    new_progress = int(100 * (j+1) / num_combinations)
    if new_progress > progress:
        progress_bar += "#"
        progress = new_progress
        antiprogress_bar = antiprogress_bar[:-1]
    bar = progress_bar + antiprogress_bar
    print(f"{bar}.    Sound number {j+1} of {num_combinations}. Progress: {round(100 * (j+1) / num_combinations, 2)}%")

print("Program finished execution")
