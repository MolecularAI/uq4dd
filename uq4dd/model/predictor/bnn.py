
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class BNN(nn.Module):
    """
    Re-implementation of the BNN in Bayes by Backprop (Blundell et al., 2015).
    Author: Emma Svensson
    """

    def __init__(
        self, 
        name,
        input_dim: int, 
        hidden_dim: int, 
        output_dim: int, 
        layers: int, 
        prior_mu: int, 
        prior_rho: int,
        prior_sig: float,
    ):
        super(BNN, self).__init__()
        self.name = name
        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim 
        self.output_dim = output_dim 
        self.n_layers = layers 
        self.prior_mu = prior_mu
        self.prior_rho = prior_rho
        self.prior_sig = prior_sig
                    
        hidden_layers = [input_dim]
        for i in range(layers):
            hidden_layers.append(hidden_dim)
        hidden_layers.append(output_dim)

        self.layers = nn.ModuleList()
        for layer in range(len(hidden_layers)-2):
            self.layers.append(
                BayesWeightLayer(hidden_layers[layer], hidden_layers[layer+1], prior_mu, prior_rho, prior_sig, activation='relu')
            )

        self.layers.append(
            BayesWeightLayer(hidden_layers[-2], hidden_layers[-1], prior_mu, prior_rho, prior_sig, activation='none')
        )

    def forward(self, x):
        net_kl = 0
        for layer in self.layers:
            x, layer_kl = layer(x)
            net_kl += layer_kl
        return x, net_kl


class BayesWeightLayer(nn.Module):
    '''
    Layer used in BNN, heavily inspired by the implementation in 
    https://github.com/JavierAntoran/Bayesian-Neural-Networks/tree/master.
    '''

    def __init__(self, input_dim, output_dim, prior_mu, prior_rho, prior_sig, activation):
        super(BayesWeightLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.prior_mu = prior_mu
        self.prior_rho = prior_rho
        self.prior_sig = prior_sig
        
        # Instantiate a large Gaussian block to sample from, much faster than generating random sample every time
        self._gaussian_block = np.random.randn(10000)
        self._Var = lambda x: Variable(torch.from_numpy(x).type(torch.FloatTensor))

        # Learnable parameters
        self.W_mu = nn.Parameter(torch.Tensor(input_dim, output_dim).uniform_(-0.01, 0.01))
        self.W_rho = nn.Parameter(torch.Tensor(input_dim, output_dim).uniform_(-3, -3))

        self.b_mu = nn.Parameter(torch.Tensor(output_dim).uniform_(-0.01, 0.01))
        self.b_rho = nn.Parameter(torch.Tensor(output_dim).uniform_(-3, -3))
        
        assert activation in ['relu', 'none'], f'Activation {activation} has not been implemented for BayesLayers.'
        self.activation = nn.ReLU() if activation == 'relu' else None
    
    def forward(self, x):
        
         # calculate std
        std_w = 1e-6 + F.softplus(self.W_rho)
        std_b = 1e-6 + F.softplus(self.b_rho)

        act_W_mu = torch.mm(x, self.W_mu)  # self.W_mu + std_w * eps_W
        act_W_std = torch.sqrt(torch.mm(x.pow(2), std_w.pow(2)))
    
        eps_W = self._random(act_W_std.shape).to(x.device)
        eps_b = self._random(std_b.shape).to(x.device)

        act_W_out = act_W_mu + act_W_std * eps_W  # (batch_size, n_output)
        act_b_out = self.b_mu + std_b * eps_b

        output = act_W_out + act_b_out.unsqueeze(0).expand(x.shape[0], -1)
        output = self.activation(output) if self.activation else output
        
        if not self.training: 
            return output, 0
        
        kld = BayesWeightLayer.KLD_cost(mu_p=0, sig_p=self.prior_sig, mu_q=self.W_mu, sig_q=std_w) + \
            BayesWeightLayer.KLD_cost(mu_p=0, sig_p=0.1, mu_q=self.b_mu, sig_q=std_b)
        
        return output, kld
    
    def _random(self, shape):
        n_eps = shape[0] * shape[1] if len(shape) > 1 else shape[0]
        eps = np.random.choice(self._gaussian_block, size=n_eps)
        eps = np.expand_dims(eps, axis=1).reshape(*shape)
        return self._Var(eps)
    
    @staticmethod
    def KLD_cost(mu_p, sig_p, mu_q, sig_q):
        KLD = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
        # https://arxiv.org/abs/1312.6114 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        return KLD

