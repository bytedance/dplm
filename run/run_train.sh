export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

task=$1

save_name_suffix=$2
save_name=${task}_${save_name_suffix}

config=config/${task}/dplm.yaml
save_path=checkpoints/${save_name}

mkdir -p ./checkpoints
mkdir -p ./tensorboard

python scripts/training.py --config ${config} \
    --model.save_path ${save_path} \
    --model_checkpoint.dirpath ${save_path} \
    --Trainer.logger tensorboard 

