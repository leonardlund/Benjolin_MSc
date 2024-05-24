import torch
import torch.nn as nn
import torch.distributions as dists
import torch.functional as F
from alive_progress import alive_bar
import numpy as np

from VAE import *


def train(vae, data, epochs, beta=1e-5, device='cuda'):
    losses = np.array([])
    opt = torch.optim.Adam(vae.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.96)
    MSELoss = nn.MSELoss(reduction='mean')
    for epoch in range(epochs):
        loss_this_epoch = 0
        with alive_bar(total=len(data)) as bar:
            for i, x in enumerate(data):
                x = x.flatten()
                x = x.to(device)
                opt.zero_grad()

                z, mu, log_var = vae.encoder.forward(x)
                x_hat = vae.decoder.forward(z)

                recon_loss = MSELoss(x_hat, x)
                kl = torch.sum(-0.5 * (1 + log_var - mu**2 - torch.exp(log_var)))
                # kl = torch.sum((sigma**2 + mu**2 - torch.exp(sigma) - 1/2))  # OLD and WRONG
                """if i % 1 == 0:
                    print(f"KL: {kl.item()}. Reconstruction loss: {recon_loss.item()}")"""
                loss = recon_loss + kl * beta
                loss.backward()
                loss_this_epoch += loss  # / len(data)
                opt.step()
                bar()
        scheduler.step()
        print(f"Epoch: {epoch + 1} out of {epochs}.      Loss = {loss_this_epoch.item()}")
        losses = np.append(losses, loss_this_epoch.cpu().detach())
    print(f"Finished Training! Losses = {losses}")
    return vae, losses
