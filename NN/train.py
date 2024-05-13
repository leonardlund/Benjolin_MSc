import torch
import torch.nn as nn
import torch.distributions as dists
import torch.functional as F

from VAE import *

def train(vae, data, epochs, beta=1e-5):
    opt = torch.optim.Adam(vae.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.96)
    MSELoss = nn.MSELoss(reduction='mean')
    for epoch in range(epochs):
        loss_this_epoch = 0
        for x in data:
            # x = x.flatten()
            opt.zero_grad()

            z, mu, sigma = vae.encoder.forward(x)
            x_hat = vae.decoder.forward(z)

            recon_loss = MSELoss(x_hat, x)
            kl = torch.sum((sigma**2 + mu**2 - torch.exp(sigma) - 1/2))

            standard_vae_loss = recon_loss + kl * beta
            # print(standard_vae_loss)
            standard_vae_loss.backward()
            loss_this_epoch += standard_vae_loss
            opt.step()
        scheduler.step()
        print(f"Epoch: {epoch + 1} out of {epochs}.      Loss = {loss_this_epoch.item()}")
    return vae