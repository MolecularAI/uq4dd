
import torch
import torchmetrics
from torchmetrics import Metric

from uq4dd.utils.loss_functions import CensoredMSELoss


def init_reg_metrics(censored): 
    if censored:
        return {
            #'MSE': torchmetrics.MeanSquaredError(),
            'CensoredMSE': CensoredMSE(),
            #'R2': torchmetrics.R2Score()
        }
    else: 
        return {
            'MSE': torchmetrics.MeanSquaredError(),
            #'R2': torchmetrics.R2Score()
        }

def init_reg_opt(censored): 
    if censored: 
        return {
            #'MSE': torchmetrics.MinMetric(),
            'CensoredMSE': torchmetrics.MinMetric(),
            #'R2': torchmetrics.MaxMetric(),
        }
    else: 
        return {
            'MSE': torchmetrics.MinMetric(),
            #'R2': torchmetrics.MaxMetric(),
        }


class CensoredMSE(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("censored_mse", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.censored_mse_loss = CensoredMSELoss(reduction='none')

    def update(self, preds, target):
        self.censored_mse += torch.sum(self.censored_mse_loss(input=preds, target=target))
        self.total += target['Label'].shape[0]

    def compute(self):
        return self.censored_mse / self.total

