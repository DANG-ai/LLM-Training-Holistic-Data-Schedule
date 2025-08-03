# Example usage:
# bash scripts/convert_to_hf.sh 3B_ods_smoothed_mean_mixed_minibatches_original_weights_init_05smoothing_seed42 30000 alon_configs/models/3B.yml

METHOD=$1
STEP=$2
MODEL_CONFIG=$3

python3 tools/convert_sequential_to_hf.py \
    --input_dir outputs/$METHOD/global_step${STEP} \
    --config_file $MODEL_CONFIG \
    --output_dir outputs/$METHOD/global_step${STEP}/hf_model