import torch
from torch import nn
from dataloader_params import ParamXYDataset

class MLPRegressor(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim):
    super(MLPRegressor, self).__init__()
    # Define layers
    self.fc1 = nn.Linear(input_dim, hidden_dim)
    self.activation1 = nn.ReLU()  # Replace with other activation if needed
    self.fc2 = nn.Linear(hidden_dim, output_dim)

  def forward(self, x):
    # Forward pass
    x = self.fc1(x)
    x = self.activation1(x)
    x = self.fc2(x)
    return x

# Example usage
model = MLPRegressor(input_dim=8, hidden_dim=4, output_dim=2)

# Define loss function and optimizer (replace with your choice)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

data_dir = ''
dataloader = ParamXYDataset(data_dir=data_dir)

# Training loop (replace with your data loading and training logic)
for epoch in range(10):
  # Forward pass, calculate loss
  outputs = model(inputs)
  loss = criterion(outputs, targets)

  # Backward pass and optimize
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  # Print training information (optional)
  print(f"Epoch: {epoch+1}, Loss: {loss.item():.4f}")

# Prediction (replace with your prediction data)
predictions = model(new_data)
