
import torch
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF


class GP():
    """
    Wrapper implementation of Sklearn GP.
    
    NOTE: Has potential for taking in ground truth aleatoric uncertainty as the alpha parameter. 
     
    Author: Emma Svensson
    """

    def __init__(self, objective, n_restarts, rbf_l, n_jobs=None):
        super(GP, self).__init__()
        
        assert objective in ['classification', 'regression'], f'Trying to make GP model for objective {objective}'
        
        # Define GP Kernel
        kernel = 1 * RBF(length_scale=rbf_l)
        
        if objective == 'classification':
            self.model = GaussianProcessClassifier(kernel=kernel, n_restarts_optimizer=n_restarts, n_jobs=n_jobs)
        else: 
            self.model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts)

    def fit(self, x, y):
        self.model.fit(x, y)
    
    def predict(self, x):
        mu, sig = self.model.predict(x, return_std=True)
        return torch.tensor(mu).unsqueeze(1), torch.tensor(sig**2).unsqueeze(1)

