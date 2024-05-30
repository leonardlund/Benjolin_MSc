from alive_progress import alive_bar
import torch
import numpy as np
from torch.cuda.amp import GradScaler


def train(vae, training_data, validation_data, epochs, opt='SGD', beta=1e-5, lr=1e-4, gamma=0.95, device='cuda'):
    training_losses = np.array([])
    validation_losses = np.array([])
    opt = torch.optim.SGD(vae.parameters(), lr=lr) if opt == 'SGD' else torch.optim.Adam(vae.parameters(), lr=lr)
    scaler = GradScaler()
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
    MSELoss = torch.nn.MSELoss(reduction='mean')
    for epoch in range(epochs):
        loss_this_epoch = 0
        with alive_bar(total=len(training_data)) as bar:
            for i, x in enumerate(training_data):
                # x = x.flatten()
                x = x.to(device)
                opt.zero_grad()

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    z, mu, log_var = vae.encoder.forward(x)
                    x_hat = vae.decoder.forward(z)
                    recon_loss = MSELoss(x_hat, x)
                    kl = torch.sum(-0.5 * (1 + log_var - mu ** 2 - torch.exp(log_var))) * beta
                    loss = recon_loss + kl

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                loss_this_epoch += loss  # / len(data)

                bar()

        scheduler.step(loss_this_epoch)
        validation_loss = 0
        for i, x in enumerate(validation_data):
            x = x.to(device)
            with torch.no_grad():
                z, mu, log_var = vae.encoder.forward(x)
                x_hat = vae.decoder.forward(z)
                recon_loss = MSELoss(x_hat, x)
                kl = torch.sum(-0.5 * (1 + log_var - mu ** 2 - torch.exp(log_var)))
                validation_loss += recon_loss + kl * beta

        print(f"Epoch: {epoch + 1} out of {epochs}. Training Loss = {loss_this_epoch.item() / len(training_data)}. "
              f"Validation Loss = {validation_loss.item() / len(validation_data)}")
        validation_losses = np.append(validation_losses, float(validation_loss.cpu().detach()))
        training_losses = np.append(training_losses, float(loss_this_epoch.cpu().detach()))
    training_losses /= len(training_data)
    validation_losses /= len(validation_data)
    print(f"Finished Training! Train = {training_losses}. Validation = {validation_losses}")
    return vae, training_losses, validation_losses
