import torch
import torch.nn as nn
import numpy as np
def box_counting(weights, box_size=3):
    """
    Computes box-counting fractal dimension by counting nonzero elements in weight patches.
    
    Parameters:
    weights (torch.Tensor): Model weight tensor
    box_size (int): Size of each box (default: 3)

    Returns:
    list: Count of nonzero elements per box
    """
    weights_flattened = weights.view(-1)
    num_boxes = len(weights_flattened) // box_size
    count = [0] * num_boxes

    for i in range(num_boxes):
        start_idx = i * box_size
        end_idx = min((i + 1) * box_size, len(weights_flattened))
        patch = weights_flattened[start_idx:end_idx]
        count[i] = torch.sum(torch.abs(patch) > 0).item()  # Convert to Python int

    return count

class ComplexFractalRegularizationLoss(nn.Module):
    """
    Custom loss function for machine learning regularization based on fractal analysis.
    
    Parameters:
    alpha (float): Scaling factor for fractal loss
    beta (float): Scaling factor for multi-resolution loss
    lambda_l1_init (float): Initial value for learnable L1 scaling factor
    resolution_factor (int): Controls depth of multi-resolution analysis
    """
    def __init__(self, alpha=0.1, beta=0.01, lambda_l1_init=0.1, resolution_factor=2):
        super(ComplexFractalRegularizationLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.lambda_l1 = nn.Parameter(torch.tensor(lambda_l1_init))  # Learnable L1 factor
        self.resolution_factor = resolution_factor  # Depth of multi-resolution

    def forward(self, model, epoch, total_epochs):
        """
        Computes the fractal regularization loss for a given model.
        
        Parameters:
        model (nn.Module): Neural network model
        epoch (int): Current training epoch
        total_epochs (int): Total training epochs

        Returns:
        torch.Tensor: Computed loss value
        """
        fractal_loss = 0
        multi_resolution_loss = 0

        # Dynamic adaptation of regularization
        progress_factor = 1 - (epoch / total_epochs)
        adjusted_alpha = self.alpha * progress_factor
        adjusted_beta = self.beta * progress_factor

        for param in model.parameters():
            if len(param.shape) > 1:  # Ignore biases
                box_counts = box_counting(param)

                # Fractal-based L1 loss with learnable scaling
                fractal_loss += torch.sum(torch.abs(param)) ** 2
                fractal_loss *= self.lambda_l1  # Dynamically scale L1

                # Multi-resolution regularization
                multi_resolution_loss += torch.mean(torch.tensor(box_counts, dtype=torch.float32)) ** self.resolution_factor

        return adjusted_alpha * fractal_loss + adjusted_beta * multi_resolution_loss
