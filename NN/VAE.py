import torch.nn as nn
import torch.distributions as dists

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, device):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.device = device

        self.dense1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.activation1 = nn.ReLU()

        self.dense2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.activation2 = nn.ReLU()
        
        self.sequential = nn.Sequential(self.dense1, self.activation1, self.dense2, self.activation2)

        self.denseMu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.denseSigma = nn.Linear(self.hidden_dim, self.latent_dim)

    def reparameterization(self, mu, sigma):
        return mu + sigma * dists.Normal(0, 1).sample(mu.shape).to(self.device)

    def forward(self, x):
        h = self.sequential(x)

        mu = self.denseMu(h)
        sigma = self.denseSigma(h)
        z = self.reparameterization(mu, sigma)
        return z, mu, sigma
    
class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, device):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.device = device

        self.dense1 = nn.Linear(self.latent_dim, self.hidden_dim)
        self.activation1 = nn.ReLU()

        self.dense2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.activation2 = nn.ReLU()

        self.dense3 = nn.Linear(self.hidden_dim, self.input_dim)
        self.activation3 = nn.Tanh()

        self.sequential = nn.Sequential(self.dense1, self.activation1, self.dense2, self.activation2, 
                                        self.dense3, self.activation3)
    
    def forward(self, z):
        x_hat = self.sequential(z)
        return x_hat


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, device='cuda'):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.device = device
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, self.device)
        self.decoder = Decoder(input_dim, hidden_dim, latent_dim, self.device)

    def forward(self, x):
        x = x.flatten()
        z, _, _ = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z
    