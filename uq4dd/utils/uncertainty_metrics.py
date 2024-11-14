
import wandb
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import bootstrap
from sklearn.linear_model import LinearRegression

import torch
import torchmetrics
from torchmetrics import Metric
from torchmetrics.functional.regression import spearman_corrcoef

from uq4dd.utils.loss_functions import CensoredMSELoss


def init_uq_metrics(censored, objective): 
    if objective == 'regression' and censored:
        return {
            'NLL': NLL(),           # NOTE: In censored mode, NLL is only computed for observed data points!
            'Censored SRCC': SRCC(objective=objective, has_censored=censored, include_censored=True),   
            'SRCC': SRCC(objective=objective, has_censored=censored, include_censored=False),
            'Censored ENCE': ENCE(objective=objective, has_censored=censored, include_censored=True, bins=20),
            'ENCE': ENCE(objective=objective, has_censored=censored, include_censored=False, bins=20),
            'CV': CV(),
            #'MiscalibrationArea': MiscalibrationArea()
        }
    return {
            'NLL': NLL(),      
            'SRCC': SRCC(objective=objective, has_censored=censored, include_censored=True),       # TODO Consider what error should be used inside the UQ evaluation for classification
            'ENCE': ENCE(objective=objective, has_censored=censored, include_censored=True, bins=20),
            'CV': CV(),
            #'MiscalibrationArea': MiscalibrationArea()
        }

def init_uq_opt(censored, objective): 
    if objective == 'regression' and censored: 
        return {
            'NLL': torchmetrics.MinMetric(),
            'Censored SRCC': torchmetrics.MaxMetric(),
            'SRCC': torchmetrics.MaxMetric(),
            'Censored ENCE': torchmetrics.MinMetric(),
            'ENCE': torchmetrics.MinMetric(),
            'CV': torchmetrics.MaxMetric(),
            #'MiscalibrationArea': torchmetrics.MinMetric()
        }
    else:
        return {
            'NLL': torchmetrics.MinMetric(),
            'SRCC': torchmetrics.MaxMetric(),
            'ENCE': torchmetrics.MinMetric(),
            'CV': torchmetrics.MaxMetric(),
            #'MiscalibrationArea': torchmetrics.MinMetric()
        }


class NLL(Metric):
    '''
    Gaussian Negative Log Likelihood proposed by Lakshminarayanan et al, 2017. 
    Additionally, provides expected performance based on 1000 simulated calculations.
    '''
    def __init__(self):
        super().__init__()
        self.add_state("uncertainty", default=torch.tensor([]), dist_reduce_fx="cat")
        self.add_state("nll", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.nll_loss = torch.nn.GaussianNLLLoss(full=True, reduction='none')

    def update(self, preds, target, uncertainty):
        assert preds.shape == uncertainty.shape
        
        if isinstance(target, dict):
            obs = target['Operator'] == 0
            target = target['Label'][obs].unsqueeze(1)
            preds = preds[obs].unsqueeze(1)
            uncertainty = uncertainty[obs].unsqueeze(1)
        
        uncertainty = uncertainty**2
        self.uncertainty = torch.cat((self.uncertainty, uncertainty))
        self.nll += torch.sum(self.nll_loss(input=preds, target=target, var=uncertainty))
        self.total += target.shape[0]

    def compute(self):
        return self.nll / self.total

    def compute_expected(self):
        # Expected NLL as proposed by Rasmussen et al., (2023)
        loss = []
        eps = torch.tensor(1e-6)
        var = self.uncertainty.to('cpu')
        for i in range(1000):
            sim_errors = torch.normal(mean=torch.zeros(var.shape), std=torch.sqrt(var))
            nll = 1/2 * (torch.log(torch.tensor(2 * np.pi)) + torch.log(torch.max(var, eps)) + sim_errors**2 / torch.max(var, eps))
            loss.append(torch.mean(nll))
        return np.mean(loss), np.std(loss)
    

class SRCC(Metric):
    '''
    Spearman's Rank Correlation Coefficient proposed by ??. 
    Additionally, provides expected performance based on 1000 simulated calculations.
    '''
    def __init__(self, objective, has_censored, include_censored):
        super().__init__()
        self.add_state("uncertainty", default=torch.tensor([]), dist_reduce_fx="cat")
        self.add_state("error", default=torch.tensor([]), dist_reduce_fx="cat")
        self.include_censored = include_censored
        if objective == 'regression' and has_censored and include_censored:
            self.L1 = CensoredMSELoss(absolute=True, reduction='none')
        elif objective == 'regression': 
            self.L1 = torch.nn.L1Loss(reduction='none')
        else: 
            self.L1 = torch.nn.BCELoss(reduction='none')

    def update(self, preds, target, uncertainty):
        
        if isinstance(target, dict) and not self.include_censored:
            obs = target['Operator'] == 0
            target = target['Label'][obs].unsqueeze(1)
            preds = preds[obs].unsqueeze(1)
            uncertainty = uncertainty[obs].unsqueeze(1)
        
        self.uncertainty = torch.cat((self.uncertainty, uncertainty))
        error = self.L1(preds, target)
        assert error.shape == uncertainty.shape
        self.error = torch.cat((self.error, error))

    def compute(self):
        corr = spearman_corrcoef(self.uncertainty[:, 0], self.error[:, 0])
        return corr
    
    def get(self):
        return self.error[:, 0].cpu(), self.uncertainty[:, 0].cpu() 
    
    def compute_expected(self):
        # Expected SRCC and NLL as proposed by Rasmussen et al., (2023)
        corr = []
        std = self.uncertainty.to('cpu')
        for i in range(1000):
            sim_errors = torch.normal(mean=torch.zeros(std.shape), std=std)
            corr.append(spearman_corrcoef(std[:, 0], sim_errors[:, 0]))
        return np.mean(corr), np.std(corr)


'''
def take_closest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.
    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0], myList[1], 0, 1
    if pos == len(myList):
        return myList[-2], myList[-1], -2, -1
    before = myList[pos - 1]
    after = myList[pos]
    if myNumber < before or myNumber > after:
        print("problem")
    else:
        return before, after, pos-1, pos


def area_function(x, observed_list, predicted_list):
    x1, x2, x1_idx, x2_idx = take_closest(predicted_list, x) 
    f = observed_list[x1_idx] + (x-x1) / (x2-x1) * (observed_list[x2_idx]-observed_list[x1_idx])
    return abs(f-x)
'''


class MiscalibrationArea(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("uncertainty", default=torch.tensor([]), dist_reduce_fx="cat")
        self.add_state("error", default=torch.tensor([]), dist_reduce_fx="cat")
        self.L1 = torch.nn.L1Loss(reduction='none')
        
    def update(self, preds, target, uncertainty):
        assert preds.shape == target.shape
        assert preds.shape == uncertainty.shape
        self.uncertainty = torch.cat((self.uncertainty, uncertainty))
        
        error = self.L1(preds, target)
        assert error.shape == uncertainty.shape
        self.error = torch.cat((self.error, error))
        
    def compute(self):
        # Implementation of miscalibration area from Rasmussen et al., (2023)
        
        # Step 1: order uncertainties and errors
        #ordered_df = pd.DataFrame()
        #ordered_df["uncertainty"] = self.uncertainty[:, 0].to('cpu')
        #ordered_df["error"] = self.error[:, 0].to('cpu')
        #ordered_df = ordered_df.sort_values(by="uncertainty")
        
        tmp = torch.cat((self.uncertainty, self.error), dim=1)
        _, ids = tmp[:, 0].sort()
        sorted_preds = tmp[ids]
        z = abs(sorted_preds[:, 1])/sorted_preds[:, 0] 
        print(z)   
        
        '''
        # Step 2: plot observed fractions of errors vs expected fraction of errors s.t. error Z = eps/sig
        pred_errors = []
        observed_errors = []
        for i in np.arange(-10, 0+0.01, 0.01):
            pred_errors.append(2*norm(loc=0, scale=1).cdf(i))
            observed_errors.append((z > abs(i)).sum()/len(z))
        
        # Step 3: integrate to get area under the curve
        area = 0
        x = min(pred_errors)
        while x < max(pred_errors):
            tmp, _ = quad(area_function, x, x+0.001, args=(observed_errors, pred_errors))
            area += tmp
            x += 0.001
        return area
        '''
        return -100


def rmse_fn(x, axis=None):
    return np.sqrt(np.mean(x**2))


class ENCE(Metric):
    '''
    Expected Normalized Calibration Error proposed by Levi et al., 2022. 
    Correspoding to Expected Calibration Error in classification. 
    Additionally, plots the error-based calibration.
    '''
    def __init__(self, objective, has_censored, include_censored, bins):
        super().__init__()
        self.add_state("uncertainty", default=torch.tensor([]), dist_reduce_fx="cat")
        self.add_state("error", default=torch.tensor([]), dist_reduce_fx="cat")
        self.include_censored = include_censored
        if objective == 'regression' and has_censored and include_censored:
            self.L1 = CensoredMSELoss(absolute=True, reduction='none')
        elif objective == 'regression': 
            self.L1 = torch.nn.L1Loss(reduction='none')
        else: 
            self.L1 = torch.nn.BCELoss(reduction='none')
        self.bins = bins
        
    def update(self, preds, target, uncertainty):
        
        if isinstance(target, dict) and not self.include_censored:
            obs = target['Operator'] == 0
            target = target['Label'][obs].unsqueeze(1)
            preds = preds[obs].unsqueeze(1)
            uncertainty = uncertainty[obs].unsqueeze(1)
        
        self.uncertainty = torch.cat((self.uncertainty, uncertainty))
        error = self.L1(preds, target)
        assert error.shape == uncertainty.shape
        self.error = torch.cat((self.error, error))
        
    def compute(self):
        rmse, rmv, ci_low, ci_high = self.compute_bins()
        rmv = torch.tensor(rmv)
        rmse = torch.tensor(rmse)
        ence = torch.mean(torch.abs(rmv-rmse)/rmv)
        return ence
    
    def compute_bins(self, confidence=False):
        error = self.error[:, 0].cpu().detach()
        uq_std = self.uncertainty[:, 0].cpu().detach() 
        
        sorted_index = torch.argsort(uq_std)
        uq_split = torch.tensor_split(uq_std[sorted_index], self.bins)
        err_split = torch.tensor_split(error[sorted_index], self.bins)
        
        rmv = []
        rmse = []
        ci_low = []
        ci_high = []
        for bin_i in range(self.bins):
            rmv.append(torch.sqrt(torch.mean(uq_split[bin_i]**2)))
            rmse.append(torch.sqrt(torch.mean(err_split[bin_i]**2))) 
            # Compute confidence interval 
            if confidence:
                res = bootstrap(err_split[bin_i].unsqueeze(0), rmse_fn, vectorized=False)
                ci_low.append(res.confidence_interval[0])
                ci_high.append(res.confidence_interval[1])
        
        return rmse, rmv, ci_low, ci_high
    
    def plot(self, wandb_logger):
        rmse, rmv, ci_low, ci_high = self.compute_bins(confidence=True)
        
        # Evaluate fit to identity function
        x = np.array(rmv).reshape(-1, 1)
        model = LinearRegression().fit(x, rmse)
        r2 = model.score(x, rmse)
        intercept = model.intercept_
        slope = model.coef_[0]
        y_pred = model.predict(x)
        
        fig, ax = plt.subplots() 
        assymetric_errors = [np.array(rmse)-np.array(ci_low), np.array(ci_high)-np.array(rmse)]
        ax.errorbar(rmv, rmse, yerr=assymetric_errors, fmt="o", linewidth=2)
        ax.plot(np.arange(rmv[0],rmv[-1],0.0001), np.arange(rmv[0],rmv[-1],0.0001), linestyle='dashed', color='k')
        ax.plot(rmv, y_pred, linestyle="dashed", color='red', label=r'$R^2$ = '+"{:0.2f}".format(r2)+", slope = {:0.2f}".format(slope)+", intercept = {:0.2f}".format(intercept))

        ax.set_xlabel("RMV") 
        ax.set_ylabel("RMSE") 
        ax.legend()
        
        wandb_logger.log({"Regression Calibration": wandb.Image(plt)})
        return fig, r2, slope, intercept


def recalibrate_uq_linear(error, uq, bins):
    sorted_index = torch.argsort(uq)
    uq_split = torch.tensor_split(uq[sorted_index], bins)
    err_split = torch.tensor_split(error[sorted_index], bins)
    
    rmv = []
    rmse = []
    for bin_i in range(bins):
        rmv.append(torch.sqrt(torch.mean(uq_split[bin_i]**2)))
        rmse.append(torch.sqrt(torch.mean(err_split[bin_i]**2))) 
    
    x = np.array(rmv).reshape(-1, 1)
    model = LinearRegression().fit(x, rmse)
        
    return model


class CV(Metric):
    '''
    STDs Coefficient of Variation proposed by Levi et al., 2022. 
    Meant as a secondary diagnostic tool after the ENCE.
    '''
    def __init__(self):
        super().__init__()
        self.add_state("uncertainty", default=torch.tensor([]), dist_reduce_fx="cat")
        
    def update(self, preds, target, uncertainty):
        self.uncertainty = torch.cat((self.uncertainty, uncertainty))
        
    def compute(self):
        std = self.uncertainty[:, 0]
        T = len(std)
        mu_s = torch.mean(std) 
        cv = torch.sqrt(torch.sum((std - mu_s) ** 2) / (T-1)) / mu_s       
        return cv

