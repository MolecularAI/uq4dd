
import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class RF():
    """
    Wrapper implementation of Sklearn RF ensemble.
    Author: Emma Svensson
    """

    def __init__(
        self,
        name, 
        objective, 
        uncertainty,
        n_estimators, 
        min_samples_split, 
        min_samples_leaf,
        max_depth,
        ccp_alpha,
        oob_score,
        verbose, 
        n_jobs
    ):
        super(RF, self).__init__()
        
        assert objective in ['classification', 'regression'], f'Trying to make RF model for objective {objective}'
        
        self.name = name
        self.uncertainty = uncertainty        
        self.n_estimators = n_estimators 
        self.min_samples_split = min_samples_split 
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.ccp_alpha = ccp_alpha
        self.oob_score = oob_score
        self.verbose = verbose 
        self.n_jobs = n_jobs
        
        if objective == 'classification':
            model_type = RandomForestClassifier
            criterion = 'gini'
            self.objective = objective
        else: 
            model_type = RandomForestRegressor
            criterion = 'squared_error'
            self.objective = objective
        
        self.model = model_type(
            n_estimators=n_estimators, 
            criterion=criterion, 
            min_samples_split=min_samples_split, 
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            ccp_alpha=ccp_alpha,
            oob_score=oob_score,
            verbose=verbose, 
            n_jobs=n_jobs
        )

    def fit(self, x, y):
        self.model.fit(x, y)
    
    def predict(self, x):
        if self.uncertainty == 'ensemble':
            mu = []
            for est in self.model.estimators_:
                if self.objective == 'classification':
                    try:
                        pred = est.predict_proba(x)[:, np.where(self.model.classes_ == 1)[0][0]]
                    except:
                        if int(est.classes_[0]) == 0:
                            pred = 1 - est.predict_proba(x).flatten()
                        elif int(est.classes_[0]) == 1:
                            pred = est.predict_proba(x).flatten()
                elif self.objective == 'regression':
                    pred = est.predict(x)
                mu.append(pred)
            mu = torch.tensor(np.stack(mu, axis=1))
            return torch.mean(mu, dim=1, keepdim=True), torch.std(mu, dim=1, keepdim=True)
        else: 
            if self.objective == 'classification':
                try:
                    pred = self.model.predict_proba(x)[:, np.where(self.model.classes_ == 1)[0][0]]
                except:
                    if int(self.model.classes_[0]) == 0:
                        pred = 1 - self.model.predict_proba(x).flatten()
                    elif int(self.model.classes_[0]) == 1:
                        pred = self.model.predict_proba(x).flatten()
            elif self.objective == 'regression':
                    pred = self.model.predict(x)
            return torch.tensor(pred).unsqueeze(1), None

