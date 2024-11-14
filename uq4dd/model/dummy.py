from typing import Any

import hydra
import wandb
import scipy
import numpy as np
import pandas as pd

import torch
from lightning import LightningModule


class DummyDTI(LightningModule):
    """
    A LightningModule for dummy predictions based on solely distribution in training data.
    """

    def __init__(
        self,
        objective,
        drug_features,
        censored: bool,
        uncertainty: str,
        recalibrate: str,
        n_predictors: int,
        predictor: str,
        verbose: bool = False
    ):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters(logger=True)
        
        assert uncertainty == 'none', 'No uncertainty supported for the dummy model.'
        assert recalibrate == 'none', 'No recalibration supported for the dummy model.'
        assert n_predictors == 1, 'Only a single predictor is supported for the dummy model.'
        if objective == 'regression':
            assert censored == False, 'Dummy model only supported for non-censored data in regression.'
        
        self.objective = objective
        self.uncertainty = uncertainty
        self.recalibrate = recalibrate
        self.n_predictors = n_predictors
        self.predictor = predictor
        self.censored = censored
        self.verbose = verbose
        
        self.train_distribution = None
    
    def evaluate(self, batch: Any, phase: str):
        x, y = batch 
        y = y['Label']    
        if self.objective == 'regression':   
            logits = torch.normal(mean=self.train_distribution['mean'], std=self.train_distribution['std'], size=y.shape)
        else: 
            logits = torch.bernoulli(torch.ones(y.shape)*self.train_distribution)
        
        # Return dummy vector as loss and uq
        return {'loss': torch.ones(y.shape), 'preds': logits, 'y': y.cpu(), 'uq': torch.zeros(y.shape)}

    def training_step(self, batch: Any, batch_idx: int):
        x, y = batch
        y = y['Label']
        assert not self.train_distribution, 'Attempting to train already fitted dummy model.'
        
        if self.objective == 'regression':
            mean, var  = scipy.stats.distributions.norm.fit(y.cpu())
            self.train_distribution = {'mean': mean, 'std': np.sqrt(var)}
        else: 
            self.train_distribution = sum(y.cpu()) / y.shape[0]
        if self.verbose:
            print(f'Train distribution {self.train_distribution}')
        
        return self.evaluate(batch, phase='train')
    
    def validation_step(self, batch: Any, batch_idx: int):
        if self.verbose:
            x, y = batch
            y = y['Label']
            if self.objective == 'regression':
                mean, var  = scipy.stats.distributions.norm.fit(y.cpu())
                tmp = {'mean': mean, 'std': np.sqrt(var)}
            else: 
                tmp = sum(y.cpu()) / y.shape[0]
            print(f'Valid distribution {tmp}')
        
        return self.evaluate(batch, phase='valid')
    
    def test_step(self, batch: Any, batch_idx: int):
        
        if self.verbose:
            x, y = batch
            y = y['Label']
            if self.objective == 'regression':
                mean, var  = scipy.stats.distributions.norm.fit(y.cpu())
                tmp = {'mean': mean, 'std': np.sqrt(var)}
            else: 
                tmp = sum(y.cpu()) / y.shape[0]
            print(f'Test distribution {tmp}')
        
        outputs = self.evaluate(batch, phase='test')
        # Temporary log of full test set
        df = pd.DataFrame({'Label': outputs['y'].cpu().numpy()[:, 0], 'Prediction': outputs['preds'].cpu().numpy()[:, 0], 'Std': outputs['uq'].cpu().numpy()[:, 0]})
        df.to_csv(f'test_dummy_{"cls" if self.objective == "classification" else "reg"}.csv')
        
        return outputs

    # Dummy optimizer as required
    def configure_optimizers(self):
        return None 

