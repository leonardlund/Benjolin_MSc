import torch.nn as nn
import torch.distributions as dists

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # self.channels, self.height, self.width = self.input_dim

        self.sequential = nn.Sequential()
        self.sequential.add_module('conv1', nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=2, padding=1))
        # self.sequential.add_module('bnorm1', nn.BatchNorm2d(num_features=2))
        self.sequential.add_module('relu1', nn.ReLU(inplace=True))
        self.sequential.add_module('pool1', nn.MaxPool2d(2, 2))
        self.sequential.add_module('conv2', nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, stride=2, padding=1))
        self.sequential.add_module('relu2', nn.ReLU(inplace=True))
        
        # TODO: don't have hardcoded dimensions here
        self.denseMu = nn.Linear(552, self.latent_dim)
        self.denseSigma = nn.Linear(552, self.latent_dim)

        
    def reparameterization(self, mu, sigma):
        return mu + sigma * dists.Normal(0, 1).sample(mu.shape)

    def forward(self, x):
        h = self.sequential(x)
        # h = h.flatten()
        h = h.view(h.shape[0], -1)
        mu = self.denseMu(h)
        sigma = self.denseSigma(h)
        z = self.reparameterization(mu, sigma)
        return z, mu, sigma
    
class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, batch_size):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size

        self.fc1 = nn.Linear(self.latent_dim, 552)
        self.reluFC = nn.ReLU()
        self.sequential = nn.Sequential()
        self.sequential.add_module('tconv2', nn.ConvTranspose2d(in_channels=4, out_channels=2, kernel_size=3, stride=2, padding=1))
        self.sequential.add_module('relu2', nn.ReLU(inplace=True))
        self.sequential.add_module('antipool', nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=3, stride=2))
        self.sequential.add_module('tconv1', nn.ConvTranspose2d(in_channels=2, out_channels=1, kernel_size=3, stride=2, padding=1))
        self.sequential.add_module('tanh1', nn.Tanh())
    
    def forward(self, z):
        h = self.reluFC(self.fc1(z))
        h = h.reshape((self.batch_size, 4, 2, 69))
        # z = z.view(-1, self.latent_dim, 1, 1)
        x_hat = self.sequential(h)
        return x_hat


class CVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, batch_size):
        super(CVAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(input_dim, hidden_dim, latent_dim, batch_size)

    def forward(self, x):
        self.decoder.batch_size = x.shape[0]
        z, _, _ = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z
    