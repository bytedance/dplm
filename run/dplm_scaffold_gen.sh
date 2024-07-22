export CUDA_VISIBLE_DEVICES=0

model_name=dplm_650m
output_dir=./generation-results/${model_name}_scaffold

mkdir -p generation-results

# Generate scaffold 
python scaffold_generate.py \
    --model_name airkingbd/${model_name} \
    --num_seqs 100 \
    --saveto $output_dir

# Predict structure by ESMFold
max_tokens=1024
pdb_path=$output_dir/scaffold_fasta/esmfold_pdb

# folding
mkdir -p $pdb_path

echo 'folding by ESMFold'
output_filename_list=$(ls ${output_dir}/scaffold_fasta)
echo $output_filename_list

python analysis/cal_plddt_dir.py -i ${output_dir}/scaffold_fasta -o ${pdb_path} --max-tokens-per-batch ${max_tokens}

echo "============================Finish Evaluation=============================="