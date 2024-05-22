import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from alive_progress import alive_bar
from dataloader_params import ParamXYDataset


class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPRegressor, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation1 = nn.ReLU()  # Replace with other activation if needed
        self.fc2 = nn.Linear(hidden_dim, 8)
        self.activation2 = nn.ReLU()
        self.fc3 = nn.Linear(8, output_dim)
        self.activation3 = nn.Sigmoid()

        self.sequential = nn.Sequential(self.fc1, self.activation1,
                                        self.fc2, self.activation2,
                                        self.fc3, self.activation3)

    def forward(self, x):
        # Forward pass
        y_hatter = self.sequential(x)
        return y_hatter


device = "cuda" if torch.cuda.is_available() else "cpu"
# Example usage
model = MLPRegressor(input_dim=2, hidden_dim=4, output_dim=8)
model.to(device)
model = model.float()

# Define loss function and optimizer (replace with your choice)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


data_dir = '/home/midml/Desktop/Leo_project/Benjolin_MA/params_dataset.npz'
# dataloader = ParamXYDataset(data_dir=data_dir)
dataset = np.load(data_dir)
targets = torch.tensor(dataset['parameter_matrix']).to(device).float()
features = torch.tensor(dataset['latent_matrix']).to(device).float()
length = features.shape[0]
print(f"Features, max value: {features.max()}. Min value: {features.min()}")
print(targets[:10, :])
epochs = 3
losses = []

"""
for epoch in range(epochs):
    loss_this_epoch = 0
    with (alive_bar(total=length) as bar):
        for i in range(length):
            # Forward pass, calculate loss
            optimizer.zero_grad()
            xi = features[i, :]
            y = (1/127) * targets[i, :]

            y_hat = model(xi) * 127
            loss = criterion(y_hat, y)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            loss_this_epoch += loss.item()
            bar()

    print(f"Epoch: {epoch + 1} out of {epochs}, Loss: {loss_this_epoch:.4f}")
    losses.append(loss_this_epoch)


save_dir = "/home/midml/Desktop/Leo_project/Benjolin_MA/NN/models/mlp_regressor-1"
torch.save(model.state_dict(), save_dir)
np.save("/home/midml/Desktop/Leo_project/Benjolin_MA/NN/models/mlp-losses-1.npy", losses)
plt.plot(losses, label="Losses over epochs")
"""