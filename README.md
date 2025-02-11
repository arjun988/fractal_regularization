# Fractal Regularization

A PyTorch-based implementation of fractal regularization for neural networks. This technique leverages box-counting and multi-resolution analysis to provide a novel approach to network regularization, potentially improving model generalization and performance.

## Installation

Install the package using pip:

```bash
pip install fractal-regularization
```

## Features

- Box-counting based fractal dimension analysis
- Multi-resolution regularization
- Learnable L1 regularization scaling
- Dynamic adaptation based on training progress
- Compatible with PyTorch neural networks
- Easy integration with existing training loops

## Usage

Here's a complete example of how to use the fractal regularization with a neural network:

```python
from fractal_reg import ComplexFractalRegularizationLoss
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

# Define your neural network
class ComplexNN(nn.Module):
    def __init__(self):
        super(ComplexNN, self).__init__()
        self.fc1 = nn.Linear(8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Initialize the loss function
fractal_loss = ComplexFractalRegularizationLoss(
    alpha=0.1,      # Scaling factor for fractal loss
    beta=0.01,      # Scaling factor for multi-resolution loss
    lambda_l1_init=0.1,  # Initial value for learnable L1 scaling
    resolution_factor=2   # Controls depth of multi-resolution analysis
)

# Training setup
model = ComplexNN()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(250):
    # Forward pass
    outputs = model(train_inputs)
    
    # Calculate main loss
    mse_loss = criterion(outputs, train_targets)
    
    # Add fractal regularization
    fractal_regularization = fractal_loss(model, epoch, 250)
    
    # Combined loss
    total_loss = mse_loss + fractal_regularization
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
```

## Parameters

The `ComplexFractalRegularizationLoss` class accepts the following parameters:

- `alpha` (float, default=0.1): Scaling factor for the fractal loss component
- `beta` (float, default=0.01): Scaling factor for the multi-resolution loss component
- `lambda_l1_init` (float, default=0.1): Initial value for the learnable L1 scaling factor
- `resolution_factor` (int, default=2): Controls the depth of multi-resolution analysis

## Example Results

When applied to a regression task using the California Housing dataset, the fractal regularization demonstrates improved model generalization compared to standard approaches:

```python
# Plot training curves
plt.plot(train_losses['Fractal'], label="Training Loss (Fractal)")
plt.plot(val_losses['Fractal'], label="Validation Loss (Fractal)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss with Fractal Regularization")
plt.legend()
plt.show()
```

## Requirements

- Python >= 3.6
- PyTorch >= 1.9.0
- NumPy >= 1.19.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.




