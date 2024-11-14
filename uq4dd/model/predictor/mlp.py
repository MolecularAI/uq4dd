import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    General implementation of simple multi-layer perceptron for single prediction.
    Author: Emma Svensson
    """

    def __init__(
        self, 
        name,
        input_dim: int, 
        hidden_dim: int, 
        output_dim: int, 
        layers: int, 
        decreasing: bool,
        dropout: float
    ):
        super(MLP, self).__init__()
        self.name = name
        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim 
        self.output_dim = output_dim 
        self.n_layers = layers 
        self.decreasing = decreasing
        self.dropout = dropout
        self.eps = 1e-6
        
        assert name in ['MLP', 'MVE', 'Evidential'], 'Fully connected neural network should be named MLP or MVE depending on the output dimension.'
        if name == 'MLP':
            assert self.output_dim == 1, f'MLPs are meant to have output dim 1 not {self.output_dim}.'
        elif name == 'Evidential':
            assert self.output_dim == 4, f'Evidental version of the MLP is meant to have output dim 4 not {self.output_dim}.'
            self.softplus = nn.Softplus()
        else: 
            assert self.output_dim == 2, f'MVE version of the MLP is intended to have output dim 2 not {self.output_dim}.'
            self.softplus = nn.Softplus()
                    
        hidden_layers = [input_dim]
        for i in range(layers):
            if decreasing:
                hidden_layers.append(int(hidden_dim/(2**i)))
            else: 
                hidden_layers.append(hidden_dim)
        hidden_layers.append(output_dim)

        inplace = False
        self.layers = nn.ModuleList()
        for layer in range(len(hidden_layers)-2):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(hidden_layers[layer], hidden_layers[layer+1]),
                    nn.ReLU(inplace=inplace),
                    nn.Dropout(p=dropout, inplace=inplace)
                )
            )

        self.layers.append(
            nn.Sequential(
                nn.Linear(hidden_layers[-2], hidden_layers[-1])
            )
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        if 'MVE' in self.name:
            mu = x[:, 0].unsqueeze(1)
            sig = self.softplus(x[:, 1].unsqueeze(1)) + self.eps
            return mu, sig
        elif self.name == 'Evidential':
            mu = x[:, 0].unsqueeze(1)
            v = self.softplus(x[:, 1].unsqueeze(1)) + self.eps
            alpha = self.softplus(x[:, 2].unsqueeze(1)) + 1 + self.eps
            beta = self.softplus(x[:, 3].unsqueeze(1)) + self.eps
            return mu, v, alpha, beta
        else: 
            return x

