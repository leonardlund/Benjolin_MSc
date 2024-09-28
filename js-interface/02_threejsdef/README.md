# Latent benjolin interface

This code opens a Web Socket in a web page that communicates with a Node.js server.
The server is responsible for relaying OSC messages bidirectionally between the web page and the python server.

## Dependencies
Check node.js and npm version:<code>node -v</code> and <code>npm -v</code>. Install node.js and npm.

## Installation

From the command line:
1. In <code>node_server</code> run <code>npm install</code>
2. In <code>frontend</code> run <code>npm install</code>
3. In <code>frontend</code> install vite
4. Install python library dependencies

## Running the Demo

1. In <code>node_server</code> run <code>node .</code>. This server will relay OSC messages from the browser to the python server through a web socket.
2. In <code>frontend</code> run <code>npx vite</code>. This script handles the 3D rendering in the browser.
3. Run the python server <code>latent_space_class-2.py</code>. This script receives coordinates from the browser and sends the corresponding benjolin parameters to the PD synthesizer. 
4. Run the PD benjolin <code>pure-data-benjo-2024.pd</code>. This pure data synthesizer receives parameters through OSC.
