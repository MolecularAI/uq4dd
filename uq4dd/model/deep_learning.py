from typing import Any

import hydra
import wandb
import numpy as np
import pandas as pd

import torch
from torch.nn import BCEWithLogitsLoss, MSELoss, GaussianNLLLoss
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric
from sklearn.linear_model import LogisticRegression

from uq4dd.model.predictor.mlp import MLP
from uq4dd.utils.loss_functions import CensoredMSELoss, TobitLoss, EvidentialLoss, CensoredEvidentialLoss
from uq4dd.utils.uncertainty_metrics import recalibrate_uq_linear
from uq4dd.utils.metrics import BestMinMetric
from uq4dd.utils.VennABERS import ScoresToMultiProbs


class DeepDTI(LightningModule):
    """
    A LightningModule for various deep learning frameworks with uncertainty quantification.
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
        self.n = n_predictors * n_experiments if self.uncertainty != 'mc' else n_experiments
        
        if self.n == 1:
            self.ensemble = predictor
        else: 
            assert ckpt_path is not None, 'Ensemble-based models can only be evaluated with pre-trained base estimators.'
            self.ensemble = torch.nn.ModuleList()
            for i in range(self.n):  
                base = MLP(    
                    name=predictor.name,
                    input_dim=predictor.input_dim, 
                    hidden_dim=predictor.hidden_dim, 
                    output_dim=predictor.output_dim, 
                    layers=predictor.n_layers, 
                    decreasing=predictor.decreasing,
                    dropout=predictor.dropout,
                )
                ckpt = torch.load(ckpt_path[i])['state_dict']
                ckpt = {key.removeprefix('ensemble.'): value for key, value in ckpt.items()}
                base.load_state_dict(ckpt, strict=False)
                self.ensemble.append(base)

        # loss functions
        if objective == 'classification':
            self.criterion = BCEWithLogitsLoss()
            assert recalibrate in ['none', 'platt_va'], f'Recalibration {recalibrate} not supported for classification.'
        elif objective == 'regression':
            if predictor.name == 'MLP':
                self.criterion = CensoredMSELoss() if censored else MSELoss()
            elif predictor.name == 'MVE': 
                self.criterion = TobitLoss() if censored else GaussianNLLLoss()
            else: 
                assert predictor.name == 'Evidential', f'Model {predictor.name} is not yet implemented.'
                self.criterion = CensoredEvidentialLoss() if censored else EvidentialLoss()
            assert recalibrate in ['none', 'uq_linear'], f'Recalibration {recalibrate} not supported for regression.'
        assert objective in ['classification', 'regression'], f'No loss function defined for objective {objective}.'
        
        self.censored = censored
        self.objective = objective
        self.delta = 1e-7
        self.recalibration_model = None
        self.save_path = save_path 
        
        # logging metrics        
        self.running_loss = torch.nn.ModuleDict({'train_loss': MeanMetric(), 'valid_loss': MeanMetric(), 'test_0_loss': MeanMetric(), 'test_1_loss': MeanMetric(), 'test_2_loss': MeanMetric()})
        self.best_loss = BestMinMetric()
        
    def forward(self, x: torch.Tensor):
        if self.uncertainty == 'mve' or self.uncertainty == 'gmm':
            if self.n == 1:
                mean, var = self.ensemble(x)
                mean = [mean]
                var = [var]
            else: 
                mean = []
                var = []
                for predictor in self.ensemble:
                    tmp = predictor(x)
                    mean.append(tmp[0])
                    var.append(tmp[1])
            return torch.cat(mean, dim=1), torch.cat(var, dim=1)
        elif self.uncertainty == 'evidential':
            if self.n == 1:
                gamma, v, alpha, beta = self.ensemble(x)
                gamma = [gamma]
                v = [v]
                alpha = [alpha]
                beta = [beta]
            else: 
                gamma = []
                v = []
                alpha = []
                beta = []
                for predictor in self.ensemble:
                    tmp = predictor(x)
                    gamma.append(tmp[0])
                    v.append(tmp[1])
                    alpha.append(tmp[2])
                    beta.append(tmp[3])
            return torch.cat(gamma, dim=1), torch.cat(v, dim=1), torch.cat(alpha, dim=1), torch.cat(beta, dim=1)
        else: 
            if self.n == 1:
                y = [self.ensemble(x)]
            else: 
                y = []
                for predictor in self.ensemble:
                    y.append(predictor(x))
            return torch.cat(y, dim=1)

    def on_train_start(self):
        self.running_loss['valid_loss'].reset()
        self.best_loss.reset()
    
    def model_step(self, batch: Any, phase: str):
        x, y = batch
        labels = y['Label']   
        logits = self.forward(x)
        
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

        if self.uncertainty == 'mve':
            mean = logits[0]
            var_al = logits[1] ** 2
            var_ep = None
            uq = logits[1]
            loss = self.criterion(mean, tmp_y, var_al) if 'test' not in phase else 0
            
        elif self.uncertainty == 'gmm':
            loss = self.criterion(logits[0], tmp_y, logits[1] ** 2) if 'test' not in phase else 0
            mu = torch.reshape(logits[0], (-1, self.n_experiments, self.n_predictors)) 
            sig = torch.reshape(logits[1], (-1, self.n_experiments, self.n_predictors))
            mean = torch.mean(mu, dim=2)
            var_al = torch.mean(sig ** 2, dim=2)
            var_ep = torch.mean(mu ** 2, dim=2) - mean ** 2 # TODO change to torch var avoid std = 0
            uq = torch.sqrt(var_al + var_ep)
            
        elif self.uncertainty == 'evidential':
            gamma = logits[0]
            v = logits[1]
            alpha = logits[2]
            beta = logits[3]
            loss = self.criterion(tmp_y, gamma, v, alpha, beta) if 'test' not in phase else 0
            mean = gamma
            var_al = beta/(alpha - 1)
            var_ep = beta/((alpha - 1) * v)
            uq = torch.sqrt(var_al + var_ep)
            
        else: 
            loss = self.criterion(logits, tmp_y) if 'test' not in phase else 0
        
            # Perform MC-Dropout during calibration and testing 
            mc_mode = phase == 'predict' or 'test' in phase
            if self.uncertainty == 'mc' and mc_mode: 
                self.ensemble.train()
                logits = [logits.unsqueeze(2)] 
                for _ in range(self.n_predictors-1):  
                    logits.append(self.forward(x).unsqueeze(2))
                logits = torch.cat(logits, dim=2)
            else: 
                n_base = self.n_predictors if self.uncertainty != 'mc' else 1
                logits = torch.reshape(logits, (-1, self.n_experiments, n_base))  

            # Post-process prediction
            if self.objective == 'classification':
                preds = torch.sigmoid(logits)
            else: 
                preds = logits

            # Collect ensamble predictions
            mean = torch.mean(preds, dim=2)
            var_ep = torch.var(preds, dim=2)
            var_al = None   # TODO derive if possible in classification
            uq = torch.std(preds, dim=2)   

        if phase != 'predict':
            self.running_loss[f'{phase}_loss'](loss)
            logger = 'test' not in phase and self.logger is not None
            self.log(f'{phase}/loss', self.running_loss[f'{phase}_loss'], on_step=False, on_epoch=True, prog_bar=True, logger=logger)
        
        return {'loss': loss, 'preds': mean, 'y': y, 'uq': uq, 'var_al': var_al, 'var_ep': var_ep}

    def training_step(self, batch: Any, batch_idx: int):
        return self.model_step(batch, phase='train')

    def on_train_epoch_end(self):
        self.running_loss[f'train_loss'].reset()
    
    def validation_step(self, batch: Any, batch_idx: int):   
        return self.model_step(batch, phase='valid')

    def on_validation_epoch_end(self):
        self.best_loss(self.running_loss[f'valid_loss'].compute(), torch.tensor(self.current_epoch))
        best_valid, _ = self.best_loss.compute()
        self.log('best_valid/loss', best_valid, sync_dist=True, on_epoch=True, logger=False)
        self.running_loss[f'valid_loss'].reset()

    def predict_step(self, batch: Any, batch_idx: int):
        
        if self.recalibrate == 'platt_va': # Platt-scaling
            outputs = self.model_step(batch, phase='predict')

            x, y = batch
            y = y['Label'].repeat(1, self.n_experiments)
            #round Labels in case of probabilistic labels            
            y = torch.round(y, decimals = 0)

            preds_preplatt = outputs['preds']
            if 1.0 in preds_preplatt:
                preds_preplatt[preds_preplatt == 1.0] = 1-self.delta
            logits_preplatt = torch.logit(preds_preplatt)

            self.recalibration_model = []
            for i in range(self.n_experiments):    
                model = LogisticRegression()
                model.fit(logits_preplatt[:,i].unsqueeze(dim=1).cpu(), y[:,i].unsqueeze(dim=1).cpu())
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
            outputs_te = self.model_step(batch, phase=f'test_{dataloader_idx}')
            
            preds_precal_te = outputs_te['preds']
            if 1.0 in preds_precal_te:
                preds_precal_te[preds_precal_te == 1.0] = 1-self.delta
            logits_precal_te = torch.logit(preds_precal_te).cpu()

            #Platt
            preds_platt = []
            for i in range(self.n_experiments):
                tmp_preds = torch.from_numpy(self.recalibration_model[i].predict_proba(logits_precal_te[:,i].reshape(-1,1))[:,torch.where(torch.from_numpy(self.recalibration_model[i].classes_ == 1))[0][0]])
                preds_platt.append(tmp_preds.unsqueeze(1))
            preds_platt = torch.cat(preds_platt, dim=1)

            #VennABERS
            """
            VennABERS based on the implementation in VennABERS.py by Paolo Toccaceli, Royal Holloway, Univ. of London.
            Implementation based on "Large-scale probabilistic prediction with and without validity guarantees" (2015).
            See https://github.com/ptocca/VennABERS  for details.
            """

            outputs_val = self.model_step(self.val_set, phase=f'test_{dataloader_idx}')

            x_val, y_val = self.val_set
            y_val = y_val['Label'].repeat(1, self.n_experiments)
            #in case of probabilistic labels
            y_val = torch.round(y_val, decimals = 0).cpu()

            preds_precal_val = outputs_val['preds']
            logits_precal_val = torch.logit(preds_precal_val).cpu()

            preds_va = []
            for i in range(self.n_experiments):
                val = list(zip(logits_precal_val[:,i].squeeze().numpy(), y_val[:,i].squeeze().numpy()))

                tmp_va = []
                probs_lower, probs_upper = ScoresToMultiProbs(val, logits_precal_te[:,i].squeeze().numpy())

                """
                Provide uncertainties based on the VennABERS interval. Based on the concept of p0, p1 discordance
                proposed in "Comparison of Scaling Methods to Obtain Calibrated Probabilities of Activity for Protein-Ligand
                Predictions." J Chem Inf Model. (2020)
                """

                for prob_lower, prob_upper in zip(probs_lower, probs_upper):
                    tmp_va.append(prob_upper / (1.0 - prob_lower + prob_upper))
                #preds_VA = torch.tensor(preds_VA).flatten()
                #preds_VA = torch.unsqueeze(preds_VA, 1)
                tmp_va = torch.tensor(tmp_va)
                tmp_va = tmp_va.reshape([len(logits_precal_te), 1])
                preds_va.append(tmp_va)
            preds_va = torch.cat(preds_va, dim=1)


            if self.save_path:
                label_te = torch.round(outputs_te['y']['Label'], decimals = 0).cpu().numpy()[:, 0]
                label_te_prob = outputs_te['y']['Label'].cpu().numpy()[:, 0]
                prob_labels = True if len(np.unique(label_te_prob)) > 2 else False
                df = {
                    'Label': label_te,
                    'Prob Label': label_te_prob if prob_labels else None
                }
                if self.n_experiments > 1:
                    for i in range(self.n_experiments):
                        df[f'Prediction_{i}'] = outputs_te['preds'].cpu().numpy()[:, i] 
                        df[f'Std_{i}'] = outputs_te['uq'].cpu().numpy()[:, i]
                        df[f'Platt_{i}'] = preds_platt.cpu().numpy()[:, i] 
                        df[f'VA_{i}'] = preds_va.cpu().numpy()[:, i]
                
                else: 
                    df['Prediction'] = outputs['preds'].cpu().numpy()[:, 0] 
                    df['Std'] = outputs['uq'].cpu().numpy()[:, 0]
                    df['Platt'] = preds_platt.cpu().numpy()[:, 0]
                    df['VA'] = preds_va.cpu().numpy()[:, 0]
                df = pd.DataFrame(df)
                df.to_csv(f'{self.save_path}_testset{4-dataloader_idx}.csv', index=False)
                outputs = outputs_te

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
                        df[f'Re Std Al_{i}'] = re_std_al.cpu().numpy()[:, i] if re_std_al  is not None else None
                        df[f'Std Ep_{i}'] = std_ep.cpu().numpy()[:, i] if std_ep  is not None else None
                        df[f'Re Std Ep_{i}'] = re_std_ep.cpu().numpy()[:, i] if re_std_ep  is not None else None
                else: 
                    df['Prediction'] = outputs['preds'].cpu().numpy()[:, 0] 
                    df['Std'] = outputs['uq'].cpu().numpy()[:, 0]
                    df['Re Std'] = re_std.cpu().numpy()[:, 0]
                    df['Std Al'] = std_al.cpu().numpy()[:, 0] if std_al  is not None else None
                    df['Re Std Al'] = re_std_al.cpu().numpy()[:, 0] if re_std_al  is not None else None
                    df['Std Ep'] = std_ep.cpu().numpy()[:, 0] if std_ep  is not None else None
                    df['Re Std Ep'] = re_std_ep.cpu().numpy()[:, 0] if re_std_ep  is not None else None
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
            best_valid, best_epoch = self.best_loss.compute()
            self.logger.experiment.summary['best_valid/loss'] = best_valid
            self.logger.experiment.summary['best_valid/epoch'] = best_epoch
    
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

