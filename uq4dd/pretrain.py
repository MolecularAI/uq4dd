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

    if cfg.model.objective == 'classification':
        datamodule.setup()
        log.info(f"Instantiating model <{cfg.model._target_}>")    
        model: LightningModule = hydra.utils.instantiate(cfg.model, BCEweight = datamodule.get_BCEweight())      
    else: 
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
    
    if cfg.test:
        # Workaround to do recalibration with the predict step
        if cfg.model.recalibrate != 'none':
            trainer.predict(model=model, datamodule=datamodule, ckpt_path='best')

        trainer.test(model=model, datamodule=datamodule, ckpt_path='best')
    
    return trainer


@hydra.main(version_base=None, config_path="../config", config_name="pretrain")
def main(cfg: DictConfig):
    
    assert cfg.model.predictor._target_ != 'uq4dd.model.predictor.rf.RF', 'Pretraining is intended for MLP base estimators, not RF.'
    
    if cfg.model.predictor._target_ == 'uq4dd.model.predictor.mlp.MLP':
        # Pick up optimal model hyperparameters for models that have already been optimized
        if cfg.model.predictor.name == 'MVE':
            lookup_path = 'config/optimal/mve.yaml'
        else: 
            lookup_path = 'config/optimal/mlp.yaml'

        lookup = yaml.safe_load(open(lookup_path, 'r'))
        censored = 'censored' if cfg.db.censored else 'observed'
        optimal = False
        if cfg.db.dataset in lookup.keys(): 
            optimal = True
            print('Running experiment with optimal hyperparameters: ')
            for name, value in lookup[cfg.db.dataset][censored].items():  
                param_path = name.split('.')
                cfg[param_path[0]][param_path[1]][param_path[2]] = value
                print(f'   {name} = {value}')
        assert optimal, 'No optimal hyperparameters found, run a sweep before pretraining models on this dataset / setting.'
    else: 
        censored = 'censored' if cfg.db.censored else 'observed'

    trainer = train(cfg)
    
    # write best checkpoint path 
    best_ckpt = trainer.checkpoint_callback.best_model_path
    print(f'Best Checkpoint saved at: {best_ckpt}')
    
    if cfg.model.objective == 'classification':
        BCE_type = 'weightedBCE' if cfg.model.use_BCEweight else 'baseBCE'
        label_type = 'probabilistic' if cfg.db.probabilistic_labels else 'hard'

        os.makedirs(f'config/checkpoints/{cfg.model.objective}/{cfg.db.split}/{cfg.model.predictor.name}/{censored}/{label_type}/{BCE_type}', exist_ok = True)
        ckpt_lookup = f'config/checkpoints/{cfg.model.objective}/{cfg.db.split}/{cfg.model.predictor.name}/{censored}/{label_type}/{BCE_type}/{cfg.db.dataset}.yaml'
    else:
        os.makedirs(f'config/checkpoints/{cfg.model.objective}/{cfg.db.split}/{cfg.model.predictor.name}/{censored}', exist_ok = True)
        ckpt_lookup = f'config/checkpoints/{cfg.model.objective}/{cfg.db.split}/{cfg.model.predictor.name}/{censored}/{cfg.db.dataset}.yaml'
    if os.path.exists(ckpt_lookup):
        lookup = yaml.safe_load(open(ckpt_lookup, 'r'))
        n = len(lookup.keys())
        if n == 1 and lookup[0] is None:
            n = 0
    else:
        lookup = {}
        n = 0
    lookup[n] = best_ckpt
    
    yaml.dump(lookup, open(ckpt_lookup, 'w'))

if __name__ == '__main__':
    main()
