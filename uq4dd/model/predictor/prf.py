import torch
import numpy as np
from PRF import prf


class PRF():
    """
    Re-implementation of a Probabilistic RF-Model (Reis et al., 2018).
    Author: Rosa Friesacher
    """

    def __init__(
        self,
        name, 
        objective, 
        uncertainty,
        n_estimators,
        max_features,
        use_py_gini,
        use_py_leafs,
        max_depth,
        keep_proba,
        bootstrap,
        min_py_sum_leaf,         
        n_jobs,
        criterion
    ):
        super(PRF, self).__init__()
        
        assert objective == 'classification', f'Trying to make probabistic RF model for objective {objective}'
        self.name = name
        self.objective = objective
        self.uncertainty = uncertainty
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.use_py_gini = use_py_gini 
        self.use_py_leafs = use_py_leafs
        self.max_depth = max_depth
        self.keep_proba = keep_proba
        self.bootstrap = bootstrap
        self.min_py_sum_leaf = min_py_sum_leaf      
        self.n_jobs = n_jobs
        model_type = prf
        self.criterion = criterion
        
    
        self.model = model_type(
            n_estimators = n_estimators,
            criterion = criterion,
            max_depth = max_depth
            )

    def fit(self, x, y):
        py = np.zeros([len(y),2])
        x = x.numpy()
        y = y.squeeze().numpy()
        py[:,1] = y
        py[:,0] = 1-py[:,1]
        self.model.fit(X = x, py = py)
    
    def predict(self, x):
      
        x = x.numpy()
        pred = np.array(self.model.predict_proba(x))
        pred = torch.from_numpy(np.array(pred)[:,1]).flatten()
        return pred.unsqueeze(1), None
        
            