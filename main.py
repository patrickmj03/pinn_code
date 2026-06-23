# import necessary libraries
import torch
import torch.nn as nn
import numpy as np
import scipy

# set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# check if GPU is available and set device accordingly + print it for documentation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Verwendetes Gerät: {device}")

# define the Physics-Informed Neural Network (PINN) class
class PINN(nn.Module):
    def __init__(self, layers):
        # use super() to call the constructor of the parent class nn.Module
        super().__init__()
        # use tanh because 2nd order derivatives are needed for the PDE
        self.activation = nn.Tanh()

        # create a list of linear layers based on the provided layer sizes
        self.linears = nn.ModuleList([
            nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)
        ])
    
    # define the forward pass of the neural network
    def forward(self, x):
        for i in range(len(self.linears)-1):
            x = self.activation(self.linears[i](x))
        x = self.linears[-1](x)
        return x    
    
# define the architecture of the neural network: 
layers = [3, 64, 64, 64,64, 3]
modell = PINN(layers).to(device)
print(modell)