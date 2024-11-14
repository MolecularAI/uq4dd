"""PyTorch Lightning Datamodule for datasets from TDC."""

import os
import math
import hydra
import pathlib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule

from tdc.single_pred import ADME

from uq4dd.datamodule.utils.datasets import DrugDataset


class TDCDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        dataset: str, # CLS: 'CYP3A4_Veith', REG: 'Lipophilicity_AstraZeneca', 'Solubility_AqSolDB'
        split: str,
        descriptors,
        censored: bool,
        objective: str,    
        threshold: float,
        probabilistic_labels: bool,
        control_std: bool,
        std: float,
        seed: int = 42,
        batch_size: int = 64,
        batch_size_eval: int = -1,
        num_workers: int = 0,
        persistent_workers: bool = True,
        pin_memory: bool = False,
        drop_last: bool = False,
        verbose: bool = False,        
    ):
        super().__init__()

        self.data_dir = data_dir
        self.verbose = verbose
        self.seed = seed
        self.censored = censored
        self.objective = objective
        self.threshold = threshold
        self.probabilistic_labels = probabilistic_labels
        self.control_std = control_std
        self.std = std
        self.BCEweight = None

        # Data split
        assert split in ['random'], f'The only supported split for public datasets is random, not {split}.'
        self.split = split

        # Data paths
        self.dataset = dataset
        dataset_dir = os.path.join(self.data_dir, dataset)
        self.raw_data_path = os.path.join(dataset_dir, 'raw')
        self.processed_data_path = os.path.join(dataset_dir, 'processed')
        if not os.path.exists(self.processed_data_path):
            os.makedirs(self.processed_data_path)
            
        # Data arguments
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval
        cuda = torch.cuda.is_available()
        self.num_workers = num_workers if cuda else 0
        self.persistent_workers = persistent_workers if cuda else False
        self.pin_memory = pin_memory if cuda else False
        self.drop_last = drop_last if cuda else False
        
        self.save_hyperparameters(logger=False)
        
        # Descriptors
        self.encoder = descriptors
        
        self.data_train: Optional[Dataset] = None
        self.data_valid: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        
        self.n_train_batches = None
    
    def encode(self, smiles):
        if self.encoder is not None:
            emb = self.encoder.encode(smiles)
        else: 
            emb = smiles
        return emb
    
    def setup(self, stage: Optional[str] = None) -> None:
        data = ADME(name=self.dataset, path=self.raw_data_path)
        
        #data.convert_to_log(form = 'standard')
        #if self.objective == 'classification':
        #    data.binarize(threshold=self.threshold, order='descending')
        
        split = data.get_split(method=self.split, seed=self.seed)
        
        if stage in ("fit", None) and not self.data_train and not self.data_valid:
            train_data = split['train']
            train_data['Drug'] = self.encode(train_data['Drug'])
            train_data['Operator'] = 0
            self.data_train = DrugDataset(train_data)
            self.n_train_batches = math.ceil(len(train_data) / self.batch_size)
            
            valid_data = split['valid']
            valid_data['Drug'] = self.encode(valid_data['Drug'])
            valid_data['Operator'] = 0
            self.data_valid = DrugDataset(valid_data)

            self.BCEweight = len(train_data[train_data['Y'] == False])/len(train_data[train_data['Y'] == True]) if len(train_data[train_data['Y'] == True]) != 0 else 1

        if stage in ("test", None) and not self.data_test:
            test_data = split['test']
            test_data['Drug'] = self.encode(test_data['Drug'])
            test_data['Operator'] = 0
            self.data_test = DrugDataset(test_data)

    def collate(self, batch):       
        xt = torch.Tensor()
        yt = {'Operator': torch.Tensor(), 'Label': torch.Tensor()}
        for x, op, y in batch: 
            xti = torch.tensor(x).float().unsqueeze(0)
            xt = xti if len(xt) == 0 else torch.cat((xt, xti), 0)
            opti = torch.tensor(op).float().unsqueeze(0)
            yt['Operator'] = opti if len(yt['Operator']) == 0 else torch.cat((yt['Operator'], opti), 0)
            yti = torch.tensor(y).float().unsqueeze(0)
            yt['Label'] = yti if len(yt['Label']) == 0 else torch.cat((yt['Label'], yti), 0)
        yt['Label'] = yt['Label'].unsqueeze(1)
        yt['Operator'] = yt['Operator'].unsqueeze(1)
        return xt, yt
    
    def get_BCEweight(self):
        return self.BCEweight

    def get_n_train_batches(self): 
        return self.n_train_batches

    def get_dataloader(self, dataset, bs, shuffle):
        return DataLoader(
            dataset,
            batch_size=bs,
            shuffle=shuffle,
            collate_fn=self.collate,
            persistent_workers=self.persistent_workers,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def train_dataloader(self):
        bs = len(self.data_train) if self.batch_size == -1 else self.batch_size
        return self.get_dataloader(self.data_train, bs, True)

    def val_dataloader(self):
        bs = len(self.data_valid) if self.batch_size_eval == -1 else self.batch_size_eval
        return self.get_dataloader(self.data_valid, bs, False)

    def predict_dataloader(self):
        bs = len(self.data_valid) if self.batch_size_eval == -1 else self.batch_size_eval
        return self.get_dataloader(self.data_valid, bs, False)

    def test_dataloader(self):
        bs = len(self.data_test) if self.batch_size_eval == -1 else self.batch_size_eval
        return self.get_dataloader(self.data_test, bs, False)

