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
    model: LightningModule = hydra.utils.instantiate(cfg.model, BCEweight = datamodule.get_BCEweight())      
 
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
    if cfg.train:
        trainer.fit(model=model, datamodule=datamodule)
        ckpt_path = 'best' if cfg.model.predictor._target_ != 'uq4dd.model.predictor.rf.RF' and cfg.model.predictor._target_ != 'uq4dd.model.predictor.prf.PRF' else None
    else:         
        ckpt_path = None #cfg.ckpt_path
        datamodule.setup()
        trainer.validate(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    # Workaround to do recalibration with the predict step
    if cfg.model.recalibrate != 'none':
        trainer.predict(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    if cfg.test:
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    
    return trainer


@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg: DictConfig):
    
    assert cfg.model.predictor._target_ == 'uq4dd.model.predictor.rf.RF' or cfg.model.predictor._target_ == 'uq4dd.model.predictor.prf.PRF', 'The regular uq4dd/train.py script is depricated for MLPs, used uq4dd/pretrain.py or uq4dd/eval.py depending on the application.'

    lookup_path = None
    ckpt_lookup = None
    
    # Pick up optimal model hyperparameters for models that have already been optimized
    if cfg.model.predictor._target_ == 'uq4dd.model.predictor.rf.RF':
        n_predictors = cfg.model.n_predictors
        lookup_path = 'config/optimal/rf.yaml'
        lookup = yaml.safe_load(open(lookup_path, 'r'))
        if cfg.db.dataset in lookup.keys():
            for name, value in lookup[cfg.db.dataset].items():  
                if name == 'checkpoint' or value is None:
                    continue
                param_path = name.split('.')
                cfg[param_path[0]][param_path[1]][param_path[2]] = value
        n_predictors = cfg.model.predictor.n_estimators
    elif cfg.model.predictor._target_ == 'uq4dd.model.predictor.mlp.MLP':
        n_predictors = cfg.model.n_predictors
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
        if not optimal: 
            print('No optimal hyperparameters found, TODO run sweep.')
        
        ckpt_lookup = f'config/checkpoints/{cfg.model.objective}/{cfg.db.split}/{cfg.model.predictor.name}/{censored}/{cfg.db.dataset}.yaml'
        assert os.path.exists(ckpt_lookup), f'Create a folder structure to place the trained checkpoint paths in as follows. "config/checkpoints/{cfg.model.objective}/{cfg.model.predictor.name}/{censored}/{cfg.db.dataset}.yaml". Then add "0: null" to the created yaml file.'
        lookup = yaml.safe_load(open(ckpt_lookup, 'r'))
        n = cfg.model.n_predictors * cfg.n_experiments if cfg.model.uncertainty == 'ensemble' else cfg.n_experiments
        
        if n > 1:
            n_avail = len(lookup.keys())
            if n_avail >= n:
                cfg.model.ckpt_path = list(lookup.values())[:n+1]
                cfg.train = False
            else: 
                assert False, f'Only {n_avail} base estimators are pre-trained, not enough for {cfg.model.uncertainty} with {cfg.model.n_predictors} predictors and {cfg.model.n_experiments} experiments.'
        
    if cfg.model.save_path:
        os.makedirs(f'results/{cfg.db.split}/{cfg.model.predictor.name}', exist_ok = True)
        cfg.model.save_path = f'results/{cfg.db.split}/{cfg.model.predictor.name}/{cfg.db.dataset}_{cfg.model.predictor.name}_{cfg.model.uncertainty}_{cfg.model.recalibrate}_n{n_predictors}_it{cfg.n_experiments}'
    
    trainer = train(cfg)
    
    if cfg.train and ckpt_lookup is not None and cfg.model.predictor._target_ == 'uq4dd.model.predictor.mlp.MLP':
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

