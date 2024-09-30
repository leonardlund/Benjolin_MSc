# Latent benjolin interface

This code opens a Web Socket in a web page that communicates with a Node.js server.
The server is responsible for relaying OSC messages bidirectionally between the web page and the python server.

## Dependencies
1. Install [Node.js](https://nodejs.org/en) using the installer from the website. Check node.js and npm version:<code>node -v</code> and <code>npm -v</code>. Install node.js and npm.
2. In <code>node_server</code> run <code>npm install</code>
3. In <code>frontend</code> run <code>npm install</code>
5. Install python library dependencies

## Running the Demo

1. In <code>node_server</code> run <code>node .</code>. This server will relay OSC messages from the browser to the python server through a web socket.
2. Run the python server <code>latent_space_class.py</code>. This script receives coordinates from the browser and sends the corresponding benjolin parameters to the PD synthesizer. 
3. Run the PD benjolin <code>pure-data-benjo-2024.pd</code>. This pure data synthesizer receives parameters through OSC.
4. In <code>frontend</code> run <code>npx vite</code>. This script handles the 3D rendering in the browser. The web application will be available at http://localhost:5173/.
