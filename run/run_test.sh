export CUDA_VISIBLE_DEVICES=0

task=$1

config=config/${task}/dplm.yaml
save_path=./checkpoints/${task}_dplm_650m
tensorboard_dir=./tensorboard/${task}_dplm_650m

python scripts/testing.py --config ${config} \
    --model.save_path ${save_path} \
    --model_checkpoint.dirpath ${save_path} \
    --Trainer.logger tensorboard 

