import sys

sys.path.append('.')

import yaml
import argparse
from yaml_config_override import add_arguments

from easydict import EasyDict
from utils.others import setup_seed
from utils.module_loader import *


def run(config):
    # Initialize a model
    model = load_model(config.model)
    
    # Initialize a dataset
    data_module = load_dataset(config.dataset)

    # Initialize a trainer
    trainer = load_trainer_testing(config)

    # Load best model and test performance
    if model.save_path is not None:
        if config.model.kwargs.get("use_lora", False):
            # Load LoRA model
            # config.model.kwargs.lora_config_path = model.save_path
            # model = load_model(config.model)
            trainer.test(model=model, datamodule=data_module, ckpt_path=os.path.join(model.save_path, 'best.ckpt'))
        else:
            model.load_checkpoint(model.save_path)
            trainer.test(model=model, datamodule=data_module)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help="running configurations", type=str, required=True)
    return parser.parse_args()


#def main(args):
def main():
    # with open(args.config, 'r', encoding='utf-8') as r:
    #     config = EasyDict(yaml.safe_load(r))
    # config = add_arguments(config)
    config = add_arguments()
    config = EasyDict(config)
    
    # Modify the config for using FSDP and lower CUDA memory occupation
    def modify_trainer_config(config):
        config.setting.os_environ.pop('WORLD_SIZE')
        config.setting.os_environ.pop('NODE_RANK')
        # config.Trainer.pop('num_nodes')
        config.dataset.train_lmdb = os.path.join(config.setting.ROOTDIR, config.dataset.train_lmdb)
        config.dataset.valid_lmdb = os.path.join(config.setting.ROOTDIR, config.dataset.valid_lmdb)
        config.dataset.test_lmdb = os.path.join(config.setting.ROOTDIR, config.dataset.test_lmdb)
        config.Trainer.devices = 1
        config.Trainer.max_epochs //= 4
        while config.dataset.dataloader_kwargs.batch_size > 1:
            config.Trainer.accumulate_grad_batches *= 2
            config.dataset.dataloader_kwargs.batch_size //= 2
            
    modify_trainer_config(config)
    
    if config.setting.seed:
        setup_seed(config.setting.seed)

    # set os environment variables
    for k, v in config.setting.os_environ.items():
        if v is not None and k not in os.environ:
            os.environ[k] = str(v)

        elif k in os.environ:
            # override the os environment variables
            config.setting.os_environ[k] = os.environ[k]

    # Only the root node will print the log
    # if config.setting.os_environ.NODE_RANK != 0:
    #     config.Trainer.logger = False
    
    run(config)


if __name__ == '__main__':
    # main(get_args())
    main()