import torch
import torch.nn as nn
import torch.distributions as dists
import torch.functional as F

from VAE import *

def train(vae, data, epochs, beta=1e-5):
    opt = torch.optim.Adam(vae.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.96)

    for epoch in range(epochs):
        loss_this_epoch = 0
        for x in data:
            x = x.flatten()
            opt.zero_grad()

            z, _, _ = vae.encoder.forward(x)
            x_hat = vae.decoder.forward(z)

            recon_loss = F.mse_loss(x_hat, x)

            standard_vae_loss = recon_loss + vae.encoder.kl * beta
            standard_vae_loss.backward()
            loss_this_epoch += standard_vae_loss
            opt.step()
        scheduler.step()
        print(f"Epoch: {epoch + 1} out of {epochs}.      Loss = {loss_this_epoch.item()}")
    return vae