# Representation Learning of DPLM
The code of this branch is based on the [SaProt: Protein Language Modeling with Structure-aware Vocabulary](https://github.com/westlake-repl/SaProt/tree/main), which we modified to support DPLM.

## Installation

```shell
# create conda virtual environment
env_name=dplm_repr

conda create -n ${env_name} python=3.10 pip
conda activate ${env_name}

# automatically install everything else
bash environment.sh
```

## Download datasets

We obtain the datasets provided by SaProt, where users can download from [here](https://drive.google.com/drive/folders/11dNGqPYfLE3M-Mbh4U7IQpuHxJpuRr4g?usp=sharing).

The datasets should be placed in the `LMDB ` folder.

## Training

The training script is as below. Training configuration is in the `config/${task}/dplm.yaml`, users can modify it based on their environment if necessary (such as gpu number, training epoch number, dropout rate and so on) .

```shell
# specify the task, which can be selected from
# "DeepLoc/cls2", "DeepLoc/cls10", "GO/CC", "GO/BP", "GO/MF", "EC",
# "HumanPPI", "MetalIonBinding", "Thermostability"
task=EC

config=config/${task}/dplm.yaml

mkdir -p ./checkpoints
mkdir -p ./tensorboard

python scripts/training.py --config ${config} \
    --Trainer.logger tensorboard 
```

## Evaluation

After training, users can evaluate with the following script:

```shell
export CUDA_VISIBLE_DEVICES=0

task=EC

config=config/${task}/dplm.yaml
save_path=./checkpoints/${task}_dplm_650m/best.ckpt

python scripts/testing.py --config ${config} \
    --model.save_path ${save_path} \
    --model_checkpoint.dirpath ${save_path}
```

Or you can download our pretrained checkpoint from [here](https://huggingface.co/airkingbd/dplm_representation_learning), and place it in the `./checkpoints/${task}_dplm_650m` folder and specify the `save_path` variable to this checkpoint in the evaluation script, as below:

```shell
export CUDA_VISIBLE_DEVICES=0

task=EC

config=config/${task}/dplm.yaml
save_path=./checkpoints/${task}_dplm_650m/EC_dplm_650m.ckpt
# The "EC_dplm_650m.ckpt" is downloaded from huggingface

python scripts/testing.py --config ${config} \
    --model.save_path ${save_path} \
    --model_checkpoint.dirpath ${save_path}
```



