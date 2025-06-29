# Placeholder for logic/architecture_generator.py
# logic/architecture_generator.py

import torch.nn as nn
import random

def generate_random_architecture(input_dim, output_dim):
    """
    Generate a random feedforward neural network.
    
    Parameters:
        input_dim (int): Number of input features
        output_dim (int): Number of output classes
    
    Returns:
        A PyTorch nn.Sequential model
    """
    layers = []
    num_hidden_layers = random.randint(1, 3)
    prev_dim = input_dim

    for _ in range(num_hidden_layers):
        hidden_dim = random.choice([16, 32, 64, 128])
        activation = random.choice([nn.ReLU(), nn.Tanh(), nn.Sigmoid()])
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(activation)
        prev_dim = hidden_dim

    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)
