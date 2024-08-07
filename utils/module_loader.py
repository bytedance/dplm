import os
import copy
import pytorch_lightning as pl
import datetime
import wandb

#from pytorch_lightning.callbacks import ModelCheckpoint
from .others import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from model.model_interface import ModelInterface
from dataset.data_interface import DataInterface
from pytorch_lightning.strategies import DDPStrategy, DDPFullyShardedStrategy
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping


def load_wandb(config):
    # initialize wandb
    wandb_config = config.setting.wandb_config
    wandb_logger = WandbLogger(project=wandb_config.project, config=config,
                               name=wandb_config.name,
                               settings=wandb.Settings(start_method='fork'))
    
    return wandb_logger


def load_tensorboard(config):
    # initialize tensorboard
    tensorboard_config = config.setting.tensorboard_config
    tensorboard_logger = TensorBoardLogger(**tensorboard_config)
    
    return tensorboard_logger


def load_model(config):
    # initialize model
    model_config = copy.deepcopy(config)
    kwargs = model_config.pop('kwargs')
    model_config.update(kwargs)
    return ModelInterface.init_model(**model_config)


def load_dataset(config):
    # initialize dataset
    dataset_config = copy.deepcopy(config)
    kwargs = dataset_config.pop('kwargs')
    dataset_config.update(kwargs)
    return DataInterface.init_dataset(**dataset_config)


# Initialize strategy
def load_strategy(config):
    config = copy.deepcopy(config)
    if "timeout" in config.keys():
        timeout = int(config.pop('timeout'))
        config["timeout"] = datetime.timedelta(seconds=timeout)

    return DDPStrategy(**config)

def load_strategy_fsdp(config):
    config = copy.deepcopy(config)
    if "timeout" in config.keys():
        timeout = int(config.pop('timeout'))
        config["timeout"] = datetime.timedelta(seconds=timeout)

    return DDPFullyShardedStrategy()
    
# Initialize a pytorch lightning trainer
def load_trainer(config):
    trainer_config = copy.deepcopy(config.Trainer)
    
    # Initialize wandb
    if trainer_config.logger is not False:
        #if trainer_config.logger == 'wandb':
        #    trainer_config.logger = load_wandb(config)
        #elif trainer_config.logger == 'tensorboard':
        trainer_config.logger = load_tensorboard(config)
        #else:
        #    raise NotImplementedError
    else:
        trainer_config.logger = False

    # Initialize callbacks
    callbacks = []
    if 'callbacks' in config:
        # initialize early stopping
        es_config = config.callbacks.early_stopping
        early_stopping = EarlyStopping(**es_config)
        callbacks.append(early_stopping)
    
    if 'model_checkpoint' in config:
        # initialize model checkpoint
        mc_config = config.model_checkpoint
        model_checkpoint = ModelCheckpoint(**mc_config)
        callbacks.append(model_checkpoint)
    
    # Initialize strategy
    strategy = load_strategy(trainer_config.pop('strategy'))
    # strategy = load_strategy_fsdp(trainer_config.pop('strategy'))
    return pl.Trainer(**trainer_config, strategy=strategy, callbacks=callbacks)


def load_trainer_ddp(config):
    trainer_config = copy.deepcopy(config.Trainer)
    
    # Initialize wandb
    if trainer_config.logger is not False:
        #if trainer_config.logger == 'wandb':
        #    trainer_config.logger = load_wandb(config)
        #elif trainer_config.logger == 'tensorboard':
        trainer_config.logger = load_tensorboard(config)
        #else:
        #    raise NotImplementedError
    else:
        trainer_config.logger = False

    # Initialize callbacks
    callbacks = []
    if 'callbacks' in config:
        # initialize early stopping
        es_config = config.callbacks.early_stopping
        early_stopping = EarlyStopping(**es_config)
        callbacks.append(early_stopping)
    
    if 'model_checkpoint' in config:
        # initialize model checkpoint
        mc_config = config.model_checkpoint
        model_checkpoint = ModelCheckpoint(**mc_config)
        callbacks.append(model_checkpoint)
    
    # Initialize plugins
    # plugins = load_plugins()
    
    # Initialize strategy
    # strategy = load_strategy(trainer_
    # Initialize plugins
    # plugins = load_plugins()
    
    # Initialize strategy
    strategy = load_strategy(trainer_config.pop('strategy'))
    #strategy = load_strategy_fsdp(trainer_config.pop('strategy'))
    return pl.Trainer(**trainer_config, strategy=strategy, callbacks=callbacks)


# Initialize a pytorch lightning trainer
def load_trainer_testing(config):
    trainer_config = copy.deepcopy(config.Trainer)
    
    # Initialize wandb
    if trainer_config.logger is not False:
        #if trainer_config.logger == 'wandb':
        #    trainer_config.logger = load_wandb(config)
        #elif trainer_config.logger == 'tensorboard':
        trainer_config.logger = load_tensorboard(config)
        # else:
        #     raise NotImplementedError
    else:
        trainer_config.logger = False

    # Initialize callbacks
    callbacks = []
    if 'callbacks' in config:
        # initialize early stopping
        es_config = config.callbacks.early_stopping
        early_stopping = EarlyStopping(**es_config)
        callbacks.append(early_stopping)
    # Initialize plugins
    # plugins = load_plugins()
    
    # Initialize strategy
    strategy = load_strategy(trainer_config.pop('strategy'))
    return pl.Trainer(**trainer_config, strategy=strategy, callbacks=callbacks)
