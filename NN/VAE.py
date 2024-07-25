import torch.nn as nn
import torch.distributions as dists
import torch


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, activation, device):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.device = device

        self.dense1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.activation1 = nn.Sigmoid() if activation == 'sigmoid' else nn.ReLU() if activation == 'relu' else nn.Tanh()

        self.dense2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.activation2 = nn.Sigmoid() if activation == 'sigmoid' else nn.ReLU() if activation == 'relu' else nn.Tanh()

        self.dense3 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.activation3 = nn.Sigmoid() if activation == 'sigmoid' else nn.ReLU() if activation == 'relu' else nn.Tanh()

        self.sequential = nn.Sequential(self.dense1, self.activation1, self.dense2, self.activation2,
                                        self.dense3, self.activation3)

        self.denseMu = nn.Linear(self.hidden_dim // 2, self.latent_dim)
        self.denseLogVar = nn.Linear(self.hidden_dim // 2, self.latent_dim)

    def reparameterization(self, mu, log_variance):
        sigma = 0.5 * torch.exp(log_variance)
        return mu + sigma * dists.Normal(0, 1).sample(mu.shape).to(self.device)

    def forward(self, x):
        h = self.sequential(x)
        mu = self.denseMu(h)
        log_variance = self.denseLogVar(h)

        z = self.reparameterization(mu, log_variance)
        return z, mu, log_variance

    
class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, activation, device):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.device = device

        self.dense1 = nn.Linear(self.latent_dim, self.hidden_dim // 2)
        self.activation1 = nn.Sigmoid() if activation == 'sigmoid' else nn.ReLU() if activation == 'relu' else nn.Tanh()

        self.dense2 = nn.Linear(self.hidden_dim // 2, self.hidden_dim)
        self.activation2 = nn.Sigmoid() if activation == 'sigmoid' else nn.ReLU() if activation == 'relu' else nn.Tanh()

        self.dense3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.activation3 = nn.Sigmoid() if activation == 'sigmoid' else nn.ReLU() if activation == 'relu' else nn.Tanh()

        self.dense4 = nn.Linear(self.hidden_dim, self.input_dim)
        self.activation4 = nn.ReLU()

        self.sequential = nn.Sequential(self.dense1, self.activation1, self.dense2, self.activation2, 
                                        self.dense3, self.activation3, self.dense4) #, self.activation4)
    
    def forward(self, z):
        x_hat = self.sequential(z)
        return x_hat


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, activation='sigmoid', device='cuda'):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.device = device
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, activation, self.device)
        self.decoder = Decoder(input_dim, hidden_dim, latent_dim, activation, self.device)

    def forward(self, x):
        x = x.flatten()
        z, _, _ = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z
    
