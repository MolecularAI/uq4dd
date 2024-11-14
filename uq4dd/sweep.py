from typing import List, Optional, Tuple

import os
import yaml
import hydra
import pyrootutils
import numpy as np
import lightning as L
from lightning import Trainer, Callback, LightningModule, LightningDataModule
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from uq4dd.utils import pylogger, instantiators, loss_functions, classification_metrics, uncertainty_metrics
from uq4dd.utils.wandb import WatchModel
from uq4dd.utils.callbacks import ClassificationMetrics, RegressionMetrics, UncertaintyMetrics
from uq4dd.datamodule.tdc import TDCDataModule
from uq4dd.datamodule.utils.descriptors import HandcraftedDescriptors
from uq4dd.model.deep_learning import DeepDTI
from uq4dd.model.predictor.mlp import MLP

log = pylogger.get_pylogger(__name__)

def train(cfg: DictConfig):

    # If no seed is choosen in the configuration, pick at random
    if cfg.seed is None or cfg.seed == -1:
        cfg.seed = np.random.randint(1, 10000)
    
    # Set random seed
    L.seed_everything(cfg.seed, workers=True)
    
    # Debug configurations
    #print(OmegaConf.to_yaml(cfg))
    
    # initialize
    log.info(f"Instantiating datamodule <{cfg.db._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.db)
    
    datamodule.setup()
    n_batches = datamodule.get_n_train_batches()
    cfg.n_train_batches = n_batches
    if cfg.trainer.log_every_n_steps > n_batches: 
        cfg.trainer.log_every_n_steps = n_batches
    
    log.info(f"Instantiating model <{cfg.model._target_}>")    
    model: LightningModule = hydra.utils.instantiate(cfg.model)      
    
    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiators.instantiate_callbacks(cfg.get("callbacks"))
    
    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiators.instantiate_loggers(cfg.get("logger"))
    
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
    
    if logger:
        log.info("Logging hyperparameters!")
        for logger in trainer.loggers:
            logger.log_hyperparams(OmegaConf.to_container(cfg))

    # train
    trainer.fit(model=model, datamodule=datamodule)

    
@hydra.main(version_base=None, config_path="../config", config_name="sweep")
def main(cfg: DictConfig):
    train(cfg)

if __name__ == '__main__':
    main()

