# Latent benjolin interface

This code opens a Web Socket in a web page that communicates with a Node.js server.
The server is responsible for relaying OSC messages bidirectionally between the web page and the python server.

## Installation

From the command line:
1. Run <code>npm install</code>
2. In the <code>web</code> folder, run <code>npm install</code>
3. Run the python server <code>latent_space_class.py</code>
4. Run the PD benjolin <code>pure-data-benjo-2024.pd</code>

## Running the Demo

1. In the main folder, start the Node.js server: <code>node .</code>
2. In <code>web</code> folder, open index.html in a web browser; a log message will be printed to the terminal when you have connected.