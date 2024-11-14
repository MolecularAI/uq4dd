

from typing import Any

import hydra
import wandb
import numpy as np
import pandas as pd

import torch
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric
from torch.nn import BCEWithLogitsLoss, MSELoss
from sklearn.linear_model import LogisticRegression

from uq4dd.model.predictor.bnn import BNN
from uq4dd.utils.loss_functions import BayesLoss, CensoredMSELoss
from uq4dd.utils.uncertainty_metrics import recalibrate_uq_linear
from uq4dd.utils.VennABERS import ScoresToMultiProbs


class BayesDTI(LightningModule):
    """
    A LightningModule for Bayesian deep learning frameworks with uncertainty quantification.
    """

    def __init__(
        self,
        objective,
        drug_features,
        censored: bool,
        uncertainty: str,
        recalibrate: str,
        n_predictors: int,
        n_experiments: int, 
        n_train_batches: int,
        predictor: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        save_path: str = None,     
        ckpt_path: str = None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=True, ignore=['predictor'])
        
        self.uncertainty = uncertainty
        self.recalibrate = recalibrate
        self.n_predictors = n_predictors
        self.n_experiments = n_experiments
        self.n = n_experiments
        
        if self.n == 1:
            self.ensemble = predictor
        else: 
            assert ckpt_path is not None, 'Ensemble-based models can only be evaluated with pre-trained base estimators.'
            self.ensemble = torch.nn.ModuleList()
            for i in range(self.n):  
                base = BNN(    
                    name=predictor.name,
                    input_dim=predictor.input_dim, 
                    hidden_dim=predictor.hidden_dim, 
                    output_dim=predictor.output_dim, 
                    layers=predictor.n_layers, 
                    prior_mu=predictor.prior_mu,
                    prior_rho=predictor.prior_rho,
                    prior_sig=predictor.prior_sig
                )
                ckpt = torch.load(ckpt_path[i])['state_dict']
                ckpt = {key.removeprefix('ensemble.'): value for key, value in ckpt.items()}
                base.load_state_dict(ckpt)
                self.ensemble.append(base)

        # loss functions
        if objective == 'classification':
            likelihood = BCEWithLogitsLoss(reduction='none')
            assert recalibrate in ['none', 'platt_va'], f'Recalibration {recalibrate} not supported for classification.'
        elif objective == 'regression':
            likelihood = CensoredMSELoss(reduction='none') if censored else MSELoss(reduction='none')
            assert recalibrate in ['none', 'uq_linear'], f'Recalibration {recalibrate} not supported for regression.'
        self.criterion = BayesLoss(likelihood=likelihood, n_train_batches=n_train_batches)
        
        self.censored = censored
        self.objective = objective
        self.recalibration_model = None
        self.save_path = save_path 
        
        # logging metrics        
        self.running_loss = torch.nn.ModuleDict({'train_loss': MeanMetric(), 'valid_loss': MeanMetric(), 'test_0_loss': MeanMetric(), 'test_1_loss': MeanMetric(), 'test_2_loss': MeanMetric()})
        self.best_loss = MinMetric()
        
    def forward(self, x: torch.Tensor):
        if self.training or self.n == 1:
            assert self.n == 1, f'Training of BNN is only supported for one model at a time, not {self.n}'
            y, kl = self.ensemble(x)
            y = [y]
            kl = kl    
        else: 
            y = []
            for predictor in self.ensemble:
                out, _ = predictor(x)
                y.append(out)
            kl = 0
        return torch.cat(y, dim=1), kl

    def on_train_start(self):
        self.running_loss['valid_loss'].reset()
        self.best_loss.reset()
    
    def model_step(self, batch: Any, phase: str):
        x, y = batch   
        logits, kl = self.forward(x)
        
        if self.objective == 'regression':
            if self.censored or 'test' in phase: 
                tmp_y = {'Label': y['Label'].repeat(1, self.n), 'Operator': y['Operator'].repeat(1, self.n)}
                y = {'Label': y['Label'].repeat(1, self.n_experiments), 'Operator': y['Operator'].repeat(1, self.n_experiments)}
            else:
                tmp_y = y['Label'].repeat(1, self.n)
                y = y['Label'].repeat(1, self.n_experiments)
        else: # self.objective == 'classification'
            #round Labels in case of probabilistic labels
            y = y['Label'].repeat(1, self.n_experiments)
            tmp_y= torch.round(tmp_y, decimals = 0)
                
        # Perform multiple inference steps during calibration and testing 
        inference_mode = phase == 'predict' or 'test' in phase
        if inference_mode: 
            logits = [logits.unsqueeze(2)] 
            for _ in range(self.n_predictors-1):
                tmp, _ = self.forward(x) 
                logits.append(tmp.unsqueeze(2))
            preds = torch.cat(logits, dim=2)
            loss = 0
        else: 
            loss = self.criterion(logits, tmp_y, kl)
            preds = torch.reshape(logits, (-1, self.n_experiments, 1))  

        # Post-process prediction
        if self.objective == 'classification':
            preds = torch.sigmoid(preds)

        # Collect ensamble predictions
        mean = torch.mean(preds, dim=2)
        var_ep = torch.var(preds, dim=2)
        var_al = None   # TODO derive if possible in classification
        uq = torch.std(preds, dim=2)   

        if phase != 'predict':
            self.running_loss[f'{phase}_loss'](loss)
            logger = 'test' not in phase and self.logger is not None
            self.log(f'{phase}/loss', self.running_loss[f'{phase}_loss'].compute(), on_step=False, on_epoch=True, prog_bar=True, logger=logger)
        
        return {'loss': loss, 'preds': mean, 'y': y, 'uq': uq, 'var_al': var_al, 'var_ep': var_ep}

    def training_step(self, batch: Any, batch_idx: int):
        return self.model_step(batch, phase='train')

    def on_train_epoch_end(self):
        self.running_loss[f'train_loss'].reset()
    
    def validation_step(self, batch: Any, batch_idx: int):   
        return self.model_step(batch, phase='valid')

    def on_validation_epoch_end(self):
        self.best_loss(self.running_loss[f'valid_loss'].compute())
        self.log('best_valid/loss', self.best_loss.compute(), sync_dist=True, on_epoch=True, logger=False)
        self.running_loss[f'valid_loss'].reset()

    def predict_step(self, batch: Any, batch_idx: int):
        
        if self.recalibrate == 'platt_va': # Platt-scaling
            x, y = batch
            y = y['Label'].repeat(1, self.n_experiments)
            #round Labels in case of probabilistic labels            
            y = torch.round(y, decimals = 0)

            logits = self.forward(x)

            self.recalibration_model = []
            for i in range(self.n_experiments):    
                model = LogisticRegression()
                model.fit(logits.cpu()[:,i], y.cpu()[:,i])
                self.recalibration_model.append(model)

            self.val_set = batch

        elif self.recalibrate == 'uq_linear': # Linear-scaling of std 
            
            outputs = self.model_step(batch, phase='predict')
            
            # Always recalibrate only based on observed labels
            if self.objective == 'regression' and self.censored:
                ops = outputs['y']['Operator'][:, 0]
                y = outputs['y']['Label'][ops == 0, :]
                preds = outputs['preds'][ops == 0, :]
                uq = outputs['uq'][ops == 0, :]
                var_ep = outputs['var_ep'][ops == 0, :] if outputs['var_ep'] is not None else None
                var_al = outputs['var_al'][ops == 0, :] if outputs['var_al'] is not None else None
            else: 
                y = outputs['y']
                preds = outputs['preds']
                uq = outputs['uq']
                var_ep = outputs['var_ep']
                var_al = outputs['var_al']
        
            L1 = torch.nn.L1Loss(reduction='none')
            error = L1(preds, y).cpu()
            uq = uq.cpu()
            var_ep = var_ep.cpu() if var_ep is not None else None
            var_al = var_al.cpu() if var_al is not None else None
            
            self.recalibration_model = {}
            
            # Total UQ
            coefficients = []
            intercepts = []
            for i in range(self.n_experiments):
                tmp_model = recalibrate_uq_linear(error[:, i], uq[:, i], bins=20)
                coefficients.append(tmp_model.coef_[0])
                intercepts.append(tmp_model.intercept_)
            
            self.recalibration_model['uq'] = {
                'coefficients': torch.tensor(coefficients).reshape(1, -1), 
                'intercepts': torch.tensor(intercepts).reshape(1, -1)
            }        
            
            # Recalibrate aleatoric part if any
            if var_al is not None: 
                coefficients = []
                intercepts = []
                for i in range(self.n_experiments):
                    tmp_model = recalibrate_uq_linear(error[:, i], torch.sqrt(var_al[:, i]), bins=20)
                    coefficients.append(tmp_model.coef_[0])
                    intercepts.append(tmp_model.intercept_)
                
                self.recalibration_model['var_al'] = {
                    'coefficients': torch.tensor(coefficients).reshape(1, -1), 
                    'intercepts': torch.tensor(intercepts).reshape(1, -1)
                }   
            else: 
                self.recalibration_model['var_al'] = None
            
            # Recalibration epistemic part if any
            if var_ep is not None: 
                coefficients = []
                intercepts = []
                for i in range(self.n_experiments):
                    tmp_model = recalibrate_uq_linear(error[:, i], torch.sqrt(var_ep[:, i]), bins=20)
                    coefficients.append(tmp_model.coef_[0])
                    intercepts.append(tmp_model.intercept_)
                
                self.recalibration_model['var_ep'] = {
                    'coefficients': torch.tensor(coefficients).reshape(1, -1), 
                    'intercepts': torch.tensor(intercepts).reshape(1, -1)
                }   
            else: 
                self.recalibration_model['var_ep'] = None
                

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        if self.recalibrate == 'platt_va':
            outputs = self.model_step(batch, phase=f'test_{dataloader_idx}')
            
            x_te, y_te = batch
            y_te = y_te['Label'].repeat(1, self.n_experiments)

            # Platt
            preds_preplatt = outputs['preds']
            logits_preplatt = torch.logit(preds_preplatt)

            preds_platt = []
            for i in range(self.n_experiments):
                tmp_preds = torch.from_numpy(self.recalibration_model[i].predict_proba(logits_preplatt.cpu()[:,i])[:,torch.where(torch.from_numpy(self.recalibration_model[i].classes_ == 1))[0][0]])
                preds_platt.append(tmp_preds.unsqueeze(1))
            preds_platt = torch.cat(preds_platt, dim=1)
            print('DEBUG: check that dim is batch_size x n_experiments')
            print(preds_platt.shape)

            # VennABERS
            """
            VennABERS based on the implementation in VennABERS.py by Paolo Toccaceli, Royal Holloway, Univ. of London.
            Implementation based on "Large-scale probabilistic prediction with and without validity guarantees" (2015).
            See https://github.com/ptocca/VennABERS  for details.
            """
            # TODO DEBUG check dimensions
            x_val, y_val = self.val_set
            y_val = y_val['Label'].repeat(1, self.n_experiments).cpu()
            # in case of probabilistic labels
            y_val = torch.round(y_val, decimals = 0)

            logits_val = self.forward(x_val).cpu()

            val = list(zip(logits_val.squeeze().numpy(), y_val.squeeze().numpy()))

            preds_preva = outputs['preds']
            logits_preva = torch.logit(preds_preva).cpu()

            preds_VA = []

            probs_lower, probs_upper = ScoresToMultiProbs(val, logits_preva.squeeze().numpy())

            """
            Provide uncertainties based on the VennABERS interval. Based on the concept of p0, p1 discordance
            proposed in "Comparison of Scaling Methods to Obtain Calibrated Probabilities of Activity for Protein-Ligand
            Predictions." J Chem Inf Model. (2020)
            """

            for prob_lower, prob_upper in zip(probs_lower, probs_upper):
                preds_VA.append(prob_upper / (1.0 - prob_lower + prob_upper))
            preds_VA = torch.tensor(preds_VA).flatten()
            preds_VA = torch.unsqueeze(preds_VA, 1)


            if self.save_path:
                # TODO save predictions (check uq_linear)
                df = pd.DataFrame({'Label': y_te.cpu().numpy()[:, 0], 'Prediction': outputs['preds'].cpu().numpy()[:, 0], 'VA Prediction': preds_VA.cpu().numpy()[:, 0],  'Platt Prediction': preds_platt.cpu().numpy()[:, 0]}) 
                df.to_csv(f'{self.save_path}_testset{4-dataloader_idx}.csv', index=False) 
        
        
        elif self.recalibrate == 'uq_linear':
            
            outputs = self.model_step(batch, phase=f'test_{dataloader_idx}')
            bs = outputs['uq'].shape[0]
            device = outputs['uq'].device
            intercept = self.recalibration_model['uq']['intercepts'].repeat(bs, 1).to(device)
            slope = self.recalibration_model['uq']['coefficients'].repeat(bs, 1).to(device)
            re_std = slope * outputs['uq'] + intercept
            
            if outputs['var_al'] is not None: 
                intercept = self.recalibration_model['var_al']['intercepts'].repeat(bs, 1).to(device)
                slope = self.recalibration_model['var_al']['coefficients'].repeat(bs, 1).to(device)
                std_al = torch.sqrt(outputs['var_al'])
                re_std_al = slope * std_al + intercept
            else:  
                std_al = None
                re_std_al = None
                
            if outputs['var_ep'] is not None: 
                intercept = self.recalibration_model['var_ep']['intercepts'].repeat(bs, 1).to(device)
                slope = self.recalibration_model['var_ep']['coefficients'].repeat(bs, 1).to(device)
                std_ep = torch.sqrt(outputs['var_ep'])
                re_std_ep = slope * std_ep + intercept
            else:  
                std_ep = None
                re_std_ep = None

            if self.save_path:      
                df = {
                    'Label': outputs['y']['Label'].cpu().numpy()[:, 0],
                    'Operator': outputs['y']['Operator'].cpu().numpy()[:, 0],
                }
                if self.n_experiments > 1:
                    for i in range(self.n_experiments):
                        df[f'Prediction_{i}'] = outputs['preds'].cpu().numpy()[:, i] 
                        df[f'Std_{i}'] = outputs['uq'].cpu().numpy()[:, i] 
                        df[f'Re Std_{i}'] = re_std.cpu().numpy()[:, i]
                        df[f'Std Al_{i}'] = std_al.cpu().numpy()[:, i] if std_al  is not None else None
                        df[f'Re Std Al_{i}'] = re_std_al.cpu().numpy()[:, i] if re_std_al is not None else None
                        df[f'Std Ep_{i}'] = std_ep.cpu().numpy()[:, i] if std_ep is not None else None
                        df[f'Re Std Ep_{i}'] = re_std_ep.cpu().numpy()[:, i] if re_std_ep is not None else None
                else: 
                    df['Prediction'] = outputs['preds'].cpu().numpy()[:, 0]  
                    df['Std'] = outputs['uq'].cpu().numpy()[:, 0]
                    df['Re Std'] = re_std.cpu().numpy()[:, 0]
                    df['Std Al'] = std_al.cpu().numpy()[:, 0] if std_al is not None else None
                    df['Re Std Al'] = re_std_al.cpu().numpy()[:, 0] if re_std_al is not None else None
                    df['Std Ep'] = std_ep.cpu().numpy()[:, 0] if std_ep is not None else None
                    df['Re Std Ep'] = re_std_ep.cpu().numpy()[:, 0] if re_std_ep is not None else None
                df = pd.DataFrame(df)
                df.to_csv(f'{self.save_path}_testset{4-dataloader_idx}.csv', index=False)
            
            outputs['uq original'] = outputs['uq']
            outputs['uq'] = re_std
            outputs['var_al original'] = outputs['var_al']
            outputs['var_al'] = re_std_al ** 2 if re_std_al is not None else None
            outputs['var_ep original'] = outputs['var_ep']
            outputs['var_ep'] = re_std_ep ** 2 if re_std_ep is not None else None
        else: 
            outputs = self.model_step(batch, phase=f'test_{dataloader_idx}')

            if self.save_path: 
                if self.objective == 'regression':     
                    std_al = torch.sqrt(outputs['var_al']) if outputs['var_al'] is not None else None
                    std_ep = torch.sqrt(outputs['var_ep']) if outputs['var_ep'] is not None else None
                    
                    df = {
                        'Label': outputs['y']['Label'].cpu().numpy()[:, 0],
                        'Operator': outputs['y']['Operator'].cpu().numpy()[:, 0],
                    }
                    if self.n_experiments > 1:
                        for i in range(self.n_experiments):
                            df[f'Prediction_{i}'] = outputs['preds'].cpu().numpy()[:, i] 
                            df[f'Std_{i}'] = outputs['uq'].cpu().numpy()[:, i] 
                            df[f'Std Al_{i}'] = std_al.cpu().numpy()[:, i] if std_al  is not None else None
                            df[f'Std Ep_{i}'] = std_ep.cpu().numpy()[:, i] if std_ep  is not None else None
                    else: 
                        df['Prediction'] = outputs['preds'].cpu().numpy()[:, 0] 
                        df['Std'] = outputs['uq'].cpu().numpy()[:, 0]
                        df['Std Al'] = std_al.cpu().numpy()[:, 0] if std_al  is not None else None
                        df['Std Ep'] = std_ep.cpu().numpy()[:, 0] if std_ep  is not None else None
                    df = pd.DataFrame(df)
                    df.to_csv(f'{self.save_path}_testset{4-dataloader_idx}.csv', index=False)
                    
                elif self.objective == 'classification':
                    print(f'Warning: Testing classification without recalibration does not support saving!')
        
        return outputs

    def on_train_end(self):
        if self.logger: 
            self.logger.experiment.summary['best_valid/loss'] = self.best_loss.compute()
    
    def configure_optimizers(self):       
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "valid/loss",  
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

