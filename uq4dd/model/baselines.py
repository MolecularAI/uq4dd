from typing import Any

import pandas as pd
import numpy as np

import torch
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric
from sklearn.linear_model import LogisticRegression

from uq4dd.model.predictor.rf import RF
from uq4dd.model.predictor.prf import PRF
from uq4dd.utils.uncertainty_metrics import recalibrate_uq_linear
from uq4dd.utils.VennABERS import ScoresToMultiProbs


class BaselineDTI(LightningModule):
    """
    A LightningModule wrapper for Sklearn models with uncertainty quantification, such as Random Forest.
    """

    def __init__(
        self,
        objective,
        drug_features,
        censored: bool,
        n_experiments: int,
        uncertainty: str,
        recalibrate: str,
        predictor,
        save_path,
        ckpt_path,
        use_BCEweight: bool,
        BCEweight,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters(logger=True)
        
        self.objective = objective
        self.uncertainty = uncertainty
        self.recalibrate = recalibrate
        self.n_experiments = n_experiments
        self.delta_platt = 1e-16
        self.predictor_name = predictor.name
        
        if self.n_experiments == 1:
            self.ensemble = predictor
            
        else: 
            self.ensemble = []
            for _ in range(self.n_experiments):
                if predictor.name == 'PRF':
                    self.ensemble.append(PRF(
                        name=predictor.name, 
                        objective=predictor.objective,
                        uncertainty=predictor.uncertainty, 
                        n_estimators=predictor.n_estimators, 
                        max_features=predictor.max_features,
                        use_py_gini=predictor.use_py_gini,
                        use_py_leafs=predictor.use_py_leafs,
                        max_depth=predictor.max_depth,
                        keep_proba=predictor.keep_proba,
                        bootstrap=predictor.bootstrap,
                        min_py_sum_leaf=predictor.min_py_sum_leaf,         
                        n_jobs=predictor.n_jobs,
                        criterion=predictor.criterion   
                    ))

                else:
                    self.ensemble.append(RF(
                        name=predictor.name, 
                        objective=predictor.objective, 
                        uncertainty=predictor.uncertainty,
                        n_estimators=predictor.n_estimators, 
                        min_samples_split=predictor.min_samples_split, 
                        min_samples_leaf=predictor.min_samples_leaf,
                        max_depth=predictor.max_depth,
                        ccp_alpha=predictor.ccp_alpha,
                        oob_score=predictor.oob_score,
                        verbose=predictor.verbose, 
                        n_jobs=predictor.n_jobs
                    ))

        if objective == 'regression':
            assert not censored, 'Sklearn RF does not allow adjusted loss for censored data.'
            self.censored = censored
        else: 
            self.censored = False       # Censored in classification doesn't affect loss functions and metrics.  
        self.recalibration_model = None
        self.save_path = save_path
        
        # logging metrics        
        self.running_loss = torch.nn.ModuleDict({'train_loss': MeanMetric(), 'valid_loss': MeanMetric(), 'test_0_loss': MeanMetric(), 'test_1_loss': MeanMetric(), 'test_2_loss': MeanMetric()})
        self.best_loss = MinMetric()
    
    def evaluate(self, batch: Any, phase: str):
        x, y = batch  
        
        if self.n_experiments > 1: 
            mean, std = self.ensemble[0].predict(x.cpu().detach())
            mean_list = [mean] 
            std_list = [std] 
            for n in range(1, self.n_experiments):
                mean, std = self.ensemble[n].predict(x.cpu().detach())
                mean_list.append(mean) 
                std_list.append(std)
            mean = torch.cat(mean_list, dim=1)
            std = torch.cat(std_list, dim=1) if self.predictor_name != 'PRF' else None
            if 'test' in phase: 
                y = {'Label': y['Label'].repeat(1, self.n_experiments), 'Operator': y['Operator'].repeat(1, self.n_experiments)}
            else: 
                y = y['Label'].repeat(1, self.n_experiments)
        else: 
            mean, std = self.ensemble.predict(x.cpu().detach())
            if 'test' in phase: 
                y = {'Label': y['Label'], 'Operator': y['Operator']}
            else: 
                y = y['Label']
        
        if self.objective == 'regression':
            if 'test' not in phase: 
                loss = torch.square((mean - y.cpu().detach()))
            else: 
                loss = torch.square((mean - y['Label'].cpu().detach()))
        else: 
            loss = torch.nn.functional.binary_cross_entropy(input=mean, target=y.cpu().detach().double(), reduction='none')

        if phase != 'predict':
            self.running_loss[f'{phase}_loss'](loss)
            logger = 'test' not in phase and self.logger is not None
            self.log(f'{phase}/loss', self.running_loss[f'{phase}_loss'].compute(), on_step=False, on_epoch=True, prog_bar=True, logger=logger)

        # Return dummy vector as loss
        return {'loss': loss, 'preds': mean, 'y': y, 'uq': std, 'var_al': None, 'var_ep': std ** 2}

    def on_train_start(self):
        self.running_loss['valid_loss'].reset()
        self.best_loss.reset()

    def training_step(self, batch: Any, batch_idx: int):
        x, y = batch
        
        y = y['Label']
        if self.n_experiments > 1: 
            for n in range(self.n_experiments):
                self.ensemble[n].fit(x.cpu().detach(), y.cpu().detach())
        else: 
            self.ensemble.fit(x.cpu().detach(), y.cpu().detach())
        
        return self.evaluate(batch, phase='train')
    
    def validation_step(self, batch: Any, batch_idx: int):   
        return self.evaluate(batch, phase='valid')

    def on_validation_epoch_end(self):
        self.best_loss(self.running_loss[f'valid_loss'].compute())
        self.log('best_valid/loss', self.best_loss.compute(), sync_dist=True, on_epoch=True, logger=False)
        self.running_loss[f'valid_loss'].reset()

    def predict_step(self, batch: Any, batch_idx: int):

        if self.recalibrate == 'platt_va': 
            x, y = batch                            # Platt-scaling
            y = y['Label']

            outputs_preplatt = self.evaluate(batch, phase = 'predict')
            preds_preplatt = outputs_preplatt['preds']
            preds_preplatt[preds_preplatt == 0.0] += self.delta_platt
            preds_preplatt[preds_preplatt == 1.0] -= self.delta_platt
            logits_preplatt = torch.logit(preds_preplatt)

            self.recalibration_model = []
            for i in range(self.n_experiments):
                model = LogisticRegression() #TODO: loop over predictors
    
                y= torch.round(y, decimals = 0)
                model.fit(logits_preplatt[:,i].unsqueeze(dim=1).cpu(), y.cpu())
                self.recalibration_model.append(model)

            self.val_set = batch                     #VennABERS
            
        elif self.recalibrate == 'uq_linear': # Linear-scaling of std 
            outputs = self.evaluate(batch, phase='predict')
            L1 = torch.nn.L1Loss(reduction='none')
            error = L1(outputs['preds'], outputs['y']).cpu() 
            uq = outputs['uq'].cpu()
            
            coefficients = []
            intercepts = []
            for n in range(self.n_experiments):
                tmp_model = recalibrate_uq_linear(error[:, n], uq[:, n], bins=20)
                coefficients.append(tmp_model.coef_[0])
                intercepts.append(tmp_model.intercept_)
            
            self.recalibration_model = {
                'coefficients': torch.tensor(coefficients).reshape(1, -1), 
                'intercepts': torch.tensor(intercepts).reshape(1, -1)
            }

    
    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):

        if self.recalibrate == 'platt_va':
            outputs_te = self.evaluate(batch, phase=f'test_{dataloader_idx}')
            x_te, y_te = batch
            y_te = y_te['Label']

            preds_precal_te = outputs_te['preds']
            preds_precal_te[preds_precal_te == 0.0] += self.delta_platt
            preds_precal_te[preds_precal_te == 1.0] -= self.delta_platt
            logits_precal_te = torch.logit(preds_precal_te).cpu()

            #Platt
            preds_platt = []
            for i in range(self.n_experiments):
                logits_precal_te_now = logits_precal_te[:,i].reshape(-1, 1)        
                preds_tmp = torch.from_numpy(self.recalibration_model[i].predict_proba(logits_precal_te_now)[:,torch.where(torch.from_numpy(self.recalibration_model[i].classes_ == 1))[0][0]])
                preds_platt.append(preds_tmp)

            #VennABERS
            """
            VennABERS based on the implementation in VennABERS.py by Paolo Toccaceli, Royal Holloway, Univ. of London.
            Implementation based on "Large-scale probabilistic prediction with and without validity guarantees" (2015).
            See https://github.com/ptocca/VennABERS  for details.
            """

            outputs_val = self.evaluate(self.val_set, phase=f'test_{dataloader_idx}')
            x_val, y_val = self.val_set
            y_val = y_val['Label']
            y_val = torch.round(y_val, decimals = 0)

            preds_precal_val = outputs_val['preds']
            logits_precal_val = torch.logit(preds_precal_val).cpu() 

            preds_va = []
            for i in range(self.n_experiments):                                    
            
                val = list(zip(logits_precal_val[:,i].squeeze().numpy(), y_val.squeeze().numpy()))

                tmp_va = []
                probs_lower, probs_upper = ScoresToMultiProbs(val, logits_precal_te[:,i].squeeze().numpy())


                """
                Provide uncertainties based on the VennABERS interval. Based on the concept of p0, p1 discordance
                proposed in "Comparison of Scaling Methods to Obtain Calibrated Probabilities of Activity for Protein-Ligand
                Predictions." J Chem Inf Model. (2020)
                """

                for prob_lower, prob_upper in zip(probs_lower, probs_upper):
                    tmp_va.append(prob_upper / (1.0 - prob_lower + prob_upper))
                tmp_va = torch.tensor(tmp_va)
                tmp_va = tmp_va.reshape([len(logits_precal_te), 1])
                preds_va.append(tmp_va)

            if self.save_path:
                label_te = torch.round(outputs_te['y'], decimals = 0).cpu().numpy()[:, 0]
                label_te_prob = outputs_te['y'].cpu().numpy()[:, 0]
                prob_labels = True if len(np.unique(label_te_prob)) > 2 else False
                df = {
                    'Label': label_te,
                    'Prob Label': label_te_prob if prob_labels else None
                }
                if self.n_experiments > 1:
                    for i in range(self.n_experiments):
                        df[f'Prediction_{i}'] = outputs_te['preds'].cpu().numpy()[:,i].reshape(-1)
                        df[f'Std_{i}'] = outputs_te['uq'].cpu().numpy()[:,i] if self.predictor_name != 'PRF' else None
                        df[f'Platt_{i}'] = preds_platt[i].reshape(-1)
                        df[f'VA_{i}'] = preds_va[i].reshape(-1)
                
                else: 
                    df['Prediction'] = outputs_te['preds'].cpu().numpy()[:,0].reshape(-1)
                    df[f'Std'] = outputs_te['uq'].cpu().numpy()[:,i] if self.predictor_name != 'PRF' else None
                    df['Platt'] = preds_platt[0].reshape(-1)
                    df['VA'] = preds_va[0].reshape(-1)
                df = pd.DataFrame(df)
                df.to_csv(f'{self.save_path}_testset{4-dataloader_idx}.csv', index=False)
                outputs = outputs_te              
                 
        elif self.recalibrate == 'uq_linear':
            outputs = self.evaluate(batch, phase=f'test_{dataloader_idx}')
            bs = outputs['uq'].shape[0]
            device = outputs['uq'].device
            intercept = self.recalibration_model['intercepts'].repeat(bs, 1).to(device)
            slope = self.recalibration_model['coefficients'].repeat(bs, 1).to(device)
            re_std = slope * outputs['uq'] + intercept
            
            if self.save_path:
                df = {
                    'Label': outputs['y']['Label'].cpu().numpy()[:, 0],
                    'Operator': outputs['y']['Operator'].cpu().numpy()[:, 0],
                }
                if self.n_experiments > 1:
                    for n in range(self.n_experiments):
                        df[f'Prediction_{n}'] = outputs['preds'].cpu().numpy()[:, n] 
                        df[f'Std_{n}'] = outputs['uq'].cpu().numpy()[:, n] 
                        df[f'Re Std_{n}'] = re_std.cpu().numpy()[:, n]
                        df[f'Std Al_{n}'] = None
                        df[f'Re Std Al_{n}'] = None
                        df[f'Std Ep_{n}'] = outputs['uq'].cpu().numpy()[:, n] 
                        df[f'Re Std Ep_{n}'] = re_std.cpu().numpy()[:, n]
                else: 
                    df['Prediction'] = outputs['preds'].cpu().numpy()[:, 0] 
                    df['Std'] = outputs['uq'].cpu().numpy()[:, 0]
                    df['Re Std'] = re_std.cpu().numpy()[:, 0]
                    df[f'Std Al'] = None
                    df[f'Re Std Al'] = None
                    df[f'Std Ep'] = outputs['uq'].cpu().numpy()[:, 0]
                    df[f'Re Std Ep'] = re_std.cpu().numpy()[:, 0]
                df = pd.DataFrame(df)
                df.to_csv(f'{self.save_path}_testset{4-dataloader_idx}.csv', index=False)
            
            outputs['uq original'] = outputs['uq']
            outputs['uq'] = re_std
            outputs['var_al original'] = None
            outputs['var_ep original'] = outputs['var_ep']
            outputs['var_ep'] = re_std ** 2
        else: 
            outputs = self.evaluate(batch, phase=f'test_{dataloader_idx}')
            
            if self.save_path: 
                if self.objective == 'regression':     
                    
                    df = {
                        'Label': outputs['y']['Label'].cpu().numpy()[:, 0],
                        'Operator': outputs['y']['Operator'].cpu().numpy()[:, 0],
                    }
                    if self.n_experiments > 1:
                        for i in range(self.n_experiments):
                            df[f'Prediction_{i}'] = outputs['preds'].cpu().numpy()[:, i] 
                            df[f'Std_{i}'] = outputs['uq'].cpu().numpy()[:, i] 
                            df[f'Std Al_{i}'] = None
                            df[f'Std Ep_{i}'] = torch.sqrt(outputs['uq']).cpu().numpy()[:, i]
                    else: 
                        df['Prediction'] = outputs['preds'].cpu().numpy()[:, 0] 
                        df['Std'] = outputs['uq'].cpu().numpy()[:, 0]
                        df['Std Al'] = None
                        df['Std Ep'] = torch.sqrt(outputs['uq']).cpu().numpy()[:, 0]
                    df = pd.DataFrame(df)
                    df.to_csv(f'{self.save_path}_testset{4-dataloader_idx}.csv', index=False)
                    
                elif self.objective == 'classification':
                    print(f'Warning: Testing classification without recalibration does not support saving!')
        
        return outputs

    def on_test_epoch_end(self):
        if self.logger:
            self.logger.experiment.summary['best_valid/loss'] = self.best_loss.compute()

    # Dummy optimizer as required
    def configure_optimizers(self):
        return None 

