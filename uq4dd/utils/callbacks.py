from typing import Any

import torch
from lightning.pytorch import Callback, Trainer, LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchmetrics import MetricCollection

from uq4dd.utils.regression_metrics import init_reg_metrics, init_reg_opt
from uq4dd.utils.classification_metrics import init_cls_metrics, init_cls_opt
from uq4dd.utils.uncertainty_metrics import init_uq_metrics, init_uq_opt


class ClassificationMetrics(Callback):
    def __init__(self, threshold: int, objective: str, probabilistic_labels = False):
        
        self.metrics = torch.nn.ModuleDict({
            'train_metrics': MetricCollection(init_cls_metrics(), prefix='train/'), 
            'valid_metrics': MetricCollection(init_cls_metrics(), prefix='valid/'), 
            'test_0_metrics': MetricCollection(init_cls_metrics(), prefix='test4/'),
            'test_1_metrics': MetricCollection(init_cls_metrics(), prefix='test3/'),
            'test_2_metrics': MetricCollection(init_cls_metrics(), prefix='test2/')
        })
        self.test_folds = []
        self.best_metrics = MetricCollection(init_cls_opt(), prefix='best_valid/')
        
        self.from_regression = objective == 'regression'
        self.from_probabilistic = probabilistic_labels == True
        self.threshold = threshold
    
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.metrics['valid_metrics'].reset()
        self.best_metrics.reset()
    
    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        preds = outputs['preds']
        y = outputs['y']
        if isinstance(y, dict):
            y = y['Label']
        
        if self.from_regression:
            preds = torch.sigmoid(preds - self.threshold)
            y = (y > self.threshold).type(torch.int)
        
        if self.from_probabilistic:
            y = (y > 0.5).type(torch.int)
        
        self.metrics[f'train_metrics'].to(pl_module.device)
        self.metrics[f'train_metrics'](preds, y)
        score = self.metrics[f'train_metrics'].compute()
        pl_module.log_dict(score, on_step=False, on_epoch=True)
    
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.metrics['train_metrics'].reset()
    
    def on_validation_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        preds = outputs['preds']
        y = outputs['y']
        if isinstance(y, dict):
            y = y['Label']
        
        if self.from_regression:
            preds = torch.sigmoid(preds - self.threshold)
            y = (y > self.threshold).type(torch.int)

        if self.from_probabilistic:
            y = (y > 0.5).type(torch.int)
        
        self.metrics[f'valid_metrics'].to(pl_module.device)
        self.metrics[f'valid_metrics'](preds, y)
        score = self.metrics[f'valid_metrics'].compute()
        pl_module.log_dict(score, on_step=False, on_epoch=True)
        
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        score = self.metrics[f'valid_metrics'].compute()
        for key, value in score.items():
            self.best_metrics[key.split("/")[1]](value)
        pl_module.log_dict(self.best_metrics.compute(), sync_dist=True, on_epoch=True, prog_bar=True, logger=False)
        self.metrics['valid_metrics'].reset() 

    def on_test_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        preds = outputs['preds']
        y = outputs['y']
        if isinstance(y, dict):
            y = y['Label']
        
        if self.from_regression:
            preds = torch.sigmoid(preds - self.threshold)
            y = (y > self.threshold).type(torch.int)

        if self.from_probabilistic:
            y = (y > 0.5).type(torch.int)
            
        self.test_folds.append(dataloader_idx)
        self.metrics[f'test_{dataloader_idx}_metrics'].to(pl_module.device)
        self.metrics[f'test_{dataloader_idx}_metrics'](preds, y)
        score = self.metrics[f'test_{dataloader_idx}_metrics'].compute()
        pl_module.log_dict(score, on_step=False, on_epoch=True, logger=False)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if pl_module.logger:
            for i in self.test_folds:
                results = self.metrics[f'test_{i}_metrics'].compute()
                for metric, score in results.items(): 
                    pl_module.logger.experiment.summary[metric] = score
                results = self.best_metrics.compute()
                for metric, score in results.items(): 
                    pl_module.logger.experiment.summary[metric] = score
            
                if 'ECE' in self.metrics[f'test_{i}_metrics'].keys():
                    self.metrics[f'test_{i}_metrics']['ECE'].plot(wandb_logger=pl_module.logger.experiment)
        
        for i in self.test_folds:
            self.metrics[f'test_{i}_metrics'].reset()

class RegressionMetrics(Callback):
    def __init__(self, objective: str, censored: bool):
        
        assert objective in ['classification', 'regression'], 'RegressionMetrics only implemented for objectives classification and regression.'
        self.on = objective == 'regression'
        
        if self.on: 
            self.metrics = torch.nn.ModuleDict({
                'train_metrics': MetricCollection(init_reg_metrics(censored=censored), prefix='train/'), 
                'valid_metrics': MetricCollection(init_reg_metrics(censored=censored), prefix='valid/'), 
                'test_0_metrics': MetricCollection(init_reg_metrics(censored=censored), prefix='test4/'),
                'test_1_metrics': MetricCollection(init_reg_metrics(censored=censored), prefix='test3/'),
                'test_2_metrics': MetricCollection(init_reg_metrics(censored=censored), prefix='test2/')
            })
            self.test_folds = []
            self.best_metrics = MetricCollection(init_reg_opt(censored=censored), prefix='best_valid/')

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.on:
            self.metrics['valid_metrics'].reset()
            self.best_metrics.reset()

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        if self.on:
            self.metrics[f'train_metrics'].to(pl_module.device)
            self.metrics[f'train_metrics'](outputs['preds'], outputs['y'])
            score = self.metrics[f'train_metrics'].compute()
            pl_module.log_dict(score, on_step=False, on_epoch=True)
    
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.on: 
            self.metrics['train_metrics'].reset()
    
    def on_validation_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        if self.on: 
            self.metrics[f'valid_metrics'].to(pl_module.device)
            self.metrics[f'valid_metrics'](outputs['preds'], outputs['y'])
            score = self.metrics[f'valid_metrics'].compute()
            pl_module.log_dict(score, on_step=False, on_epoch=True)
        
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.on:
            score = self.metrics[f'valid_metrics'].compute()
            for key, value in score.items():
                self.best_metrics[key.split("/")[1]](value)
            pl_module.log_dict(self.best_metrics.compute(), sync_dist=True, on_epoch=True, prog_bar=True, logger=False)
            self.metrics['valid_metrics'].reset() 

    def on_test_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        if self.on:
            self.test_folds.append(dataloader_idx)
            self.metrics[f'test_{dataloader_idx}_metrics'].to(pl_module.device)
            self.metrics[f'test_{dataloader_idx}_metrics'](outputs['preds'], outputs['y'])
            score = self.metrics[f'test_{dataloader_idx}_metrics'].compute()
            pl_module.log_dict(score, on_step=False, on_epoch=True, logger=False)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.on:
            if pl_module.logger:
                for i in self.test_folds:
                    results = self.metrics[f'test_{i}_metrics'].compute()
                    for metric, score in results.items(): 
                        pl_module.logger.experiment.summary[metric] = score
                    results = self.best_metrics.compute()
                    for metric, score in results.items(): 
                        pl_module.logger.experiment.summary[metric] = score
            for i in self.test_folds:
                self.metrics[f'test_{i}_metrics'].reset()
        

class UncertaintyMetrics(Callback):
    def __init__(self, objective: str, censored: bool, uncertainty: str):
        assert uncertainty in ['none', 'ensemble', 'mc', 'gaussian'], 'Uncertainty metrics only implemented for none, ensemble, mc, gaussian '

        self.censored = censored if objective == 'regression' else False
        self.uncertainty = uncertainty
        
        if self.uncertainty != 'none':
            self.metrics = torch.nn.ModuleDict({
                'train_metrics': MetricCollection(init_uq_metrics(censored=censored, objective=objective), prefix='train/'), 
                'valid_metrics': MetricCollection(init_uq_metrics(censored=censored, objective=objective), prefix='valid/'), 
                'test_0_metrics': MetricCollection(init_uq_metrics(censored=censored, objective=objective), prefix='test4/'),
                'test_1_metrics': MetricCollection(init_uq_metrics(censored=censored, objective=objective), prefix='test3/'),
                'test_2_metrics': MetricCollection(init_uq_metrics(censored=censored, objective=objective), prefix='test2/')
            })
            self.test_folds = []
            self.best_metrics = MetricCollection(init_uq_opt(censored=censored, objective=objective), prefix='best_valid/')
    
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.uncertainty != 'none':
            self.metrics['valid_metrics'].reset()
            self.best_metrics.reset()    
    
    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        
        if self.uncertainty == 'ensemble' or self.uncertainty == 'gaussian':
            self.metrics[f'train_metrics'].to(pl_module.device)
            self.metrics[f'train_metrics'](outputs['preds'], outputs['y'], outputs['uq'])
            score = self.metrics[f'train_metrics'].compute()
            pl_module.log_dict(score, on_step=False, on_epoch=True)
    
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.uncertainty != 'none':
            self.metrics['train_metrics'].reset()
    
    def on_validation_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        if self.uncertainty == 'ensemble' or self.uncertainty == 'gaussian':
            self.metrics[f'valid_metrics'].to(pl_module.device)
            self.metrics[f'valid_metrics'](outputs['preds'], outputs['y'], outputs['uq'])
            score = self.metrics[f'valid_metrics'].compute()
            pl_module.log_dict(score, on_step=False, on_epoch=True)
        
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.uncertainty == 'ensemble' or self.uncertainty == 'gaussian':
            score = self.metrics[f'valid_metrics'].compute()
            for key, value in score.items():
                self.best_metrics[key.split("/")[1]](value)
            pl_module.log_dict(self.best_metrics.compute(), sync_dist=True, on_epoch=True, prog_bar=True, logger=False)
            self.metrics['valid_metrics'].reset() 

    def on_test_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        if self.uncertainty != 'none':
            self.test_folds.append(dataloader_idx)
            self.metrics[f'test_{dataloader_idx}_metrics'].to(pl_module.device)
            self.metrics[f'test_{dataloader_idx}_metrics'](outputs['preds'], outputs['y'], outputs['uq'])
            score = self.metrics[f'test_{dataloader_idx}_metrics'].compute()
            pl_module.log_dict(score, on_step=False, on_epoch=True, logger=False)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        
        if self.uncertainty != 'none' and pl_module.logger:
            for i in self.test_folds:
                results = self.metrics[f'test_{i}_metrics'].compute()
                for metric, score in results.items(): 
                    pl_module.logger.experiment.summary[metric] = score
                expected_metrics = ['SRCC', 'Censored SRCC', 'NLL'] if self.censored else ['SRCC', 'NLL']
                for metric in expected_metrics:
                    mu, sig = self.metrics[f'test_{i}_metrics'][metric].compute_expected()
                    pl_module.logger.experiment.summary[f'test{4-i}/{metric} (Simulated mean)'] = mu
                    pl_module.logger.experiment.summary[f'test{4-i}/{metric} (Simulated std)'] = sig
                
                fig, r2, slope, intercept = self.metrics[f'test_{i}_metrics']['ENCE'].plot(wandb_logger=pl_module.logger.experiment)
                pl_module.logger.experiment.summary[f'test{4-i}/FittedR2'] = r2
                pl_module.logger.experiment.summary[f'test{4-i}/FittedSlope'] = slope
                pl_module.logger.experiment.summary[f'test{4-i}/FittedIntercept'] = intercept
                
                results = self.best_metrics.compute()
                for metric, score in results.items(): 
                    pl_module.logger.experiment.summary[metric] = score
                self.metrics[f'test_{i}_metrics'].reset()

    