export CUDA_VISIBLE_DEVICES=0

model_name=dplm_650m

python generate.py \
    --model_name airkingbd/${model_name} \
    --seq_lens 100 200 300 400 500 \
    --saveto generation-results/${model_name}

# # uncond inpainting
# python generate.py \
#     --model_name airkingbd/${model_name} \
#     --seq_lens 100 200 \
#     --saveto generation-results/${model_name}_inpainting \
#     --cond_position 1-4 8-10 \
#     --cond_seq ALVE EME
