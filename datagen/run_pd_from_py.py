"""
Example from Vincenzo
"""



import os

global pd_executable
# find the pd executable in your computer, the following works on mac
pd_executable = '/Applications/Pd-0.54-1.app/Contents/Resources/bin/pd' 
pd_script_path = 'pd_script.pd'

# pure data params entries
text_entry = 'text_text_text'
param1 = 0.1
param2 = 0.2
param3 = 0.6
param4 = 0.3

# run pure data
command = pd_executable + f' -send "; text_entry {text_entry}; param1 {param1}; param2 {param2}; param3 {param3}; param4 {param4}"  ' + pd_script_path

# include -nogui flag to execute pd without GUI
#command = pd_executable + f' -send "; filename {filename}" -nogui ' + pd_script_path

os.system(command)
