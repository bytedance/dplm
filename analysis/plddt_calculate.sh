# Example:
# bash analysis/plddt_calculate.sh generation-results/dplm_650m

output_dir=$1
output_filename_list=$(ls ${output_dir})
echo $output_filename_list

max_tokens=1024

echo "============================Begin Evaluation=============================="

pdb_path=$output_dir/esmfold_pdb

# folding
mkdir -p $pdb_path

echo 'folding by ESMFold'
python cal_plddt_dir.py -i ${output_dir} -o ${pdb_path} --max-tokens-per-batch ${max_tokens} \
	-m ${ROOTDIR}/cache-dir

echo "============================Finish Evaluation=============================="

