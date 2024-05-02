import itertools
from pythonosc import udp_client
import sounddevice as sd
import threading
from scipy.io.wavfile import write
from copy import copy
import os

global pd_executable
# find the pd executable in your computer, the following works on mac
pd_executable = '/Applications/Pd-0.54-1.app/Contents/Resources/bin/pd' 
pd_script_path = 'pd_script.pd'


duration = 2  # Duration of the recording in seconds
sample_rate = 44100  # Sample rate (samples per second)
channels = 2  # Number of audio channels (2 for stereo)

port = 7771  # must match the port declared in Pure data
client = udp_client.SimpleUDPClient("127.0.0.1", port)

# Making a lock for the threads, tbh I don't remember why I needed to handle the threads
lock = threading.Lock()

num_settings_per_knob = 4
setting_list = []
for i in range(num_settings_per_knob):
    setting_list.append((i+1)/(num_settings_per_knob+1))
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


def record_and_save(combination):
    with lock:
        client.send_message("/benjolin", combination)
        print("Sent OSC")
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels)
        print("Starting recording...")
        
        # Wait for the recording to finish
        sd.wait()
        #sd.write(filename, recording, sample_rate)
        print("Recording finished")
    
    # Generate a filename for the combination
    filename = "dataset/" + "_".join(str(param).replace('.', '') for param in combination) + ".wav"
    print(filename)
    
    write(filename, sample_rate, recording)

# Create a thread for each combination
threads = []
for combination in combinations:
    thread = threading.Thread(target=record_and_save, args=(combination,))
    thread.start()
    threads.append(thread)

# Wait for all threads to finish
for thread in threads:
    thread.join()

print("All recordings saved successfully.")
