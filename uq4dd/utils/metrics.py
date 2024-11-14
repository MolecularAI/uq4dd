

import torch
import torchmetrics
from torchmetrics import Metric


class BestMinMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('best_score', default=torch.tensor(float('inf'), dtype=torch.get_default_dtype()), dist_reduce_fx='min')
        self.add_state('best_epoch', default=torch.tensor(0), dist_reduce_fx='max')

    def update(self, value, epoch):
        if torch.min(value) <= self.best_score:
            self.best_score = torch.min(value)
            self.best_epoch = epoch

    def compute(self):
        return self.best_score, self.best_epoch

