# GENERAL CONFIGS THAT WILL BE USED FOR ALL RUNS
CONFIGS="alon_configs/data/pile.yml alon_configs/init.yml alon_configs/models/1B.yml alon_configs/eval_tasks.yml alon_configs/train_data_weights/original_pile.yml"
# Doesn't include alon_configs/parallelism.yml

WANDB_GROUP="1B_ods_smoothed_mean_mixed_minibatches_original_weights_init"
DATA_SAMPLING_METHOD="smoothed_mean"
DATA_SAMPLING_WARMUP_STEPS="2000"
DATA_SAMPLING_UPDATE_FREQUENCY="1"
MIXED_MINIBATCHES=true

# SEEDS=(1234 42 100 222)
SEEDS=(42)

# RUN SPECIFIC CONFIGS
for SEED in ${SEEDS[@]}; do
    RUN_NAME="${WANDB_GROUP}_seed${SEED}"
    ARGS="--seed ${SEED} --save outputs/${RUN_NAME} --wandb_group ${WANDB_GROUP} --wandb_run_name seed${SEED} --data_sampling_method ${DATA_SAMPLING_METHOD} --data_sampling_warmup_steps ${DATA_SAMPLING_WARMUP_STEPS} --data_sampling_update_frequency ${DATA_SAMPLING_UPDATE_FREQUENCY} --mixed_minibatches ${MIXED_MINIBATCHES}"
    python3 tools/create_run_specific_config.py ${ARGS}
    RUN_SPECIFIC_CONFIG="alon_configs/run_specific/${RUN_NAME}.yml"
    echo "Running with configs: ${CONFIGS} ${RUN_SPECIFIC_CONFIG}"
    python3 deepy.py train.py ${CONFIGS} ${RUN_SPECIFIC_CONFIG} 2>&1 | tee outputs/${RUN_NAME}.log
    
    # evaluate 0-shot
    bash scripts/evaluate.sh outputs/${RUN_NAME}/global_step100000/configs/${RUN_NAME}.yml alon_configs/models/eval_1B_1gpu.yml 2>&1 | tee outputs/${RUN_NAME}_eval.log
    # evaluate 5-shot
    NUM_SHOTS=5
    bash scripts/evaluate.sh outputs/${RUN_NAME}/global_step100000/configs/${RUN_NAME}.yml alon_configs/models/eval_1B_1gpu.yml ${NUM_SHOTS} 2>&1 | tee outputs/${RUN_NAME}_${NUM_SHOTS}shot_eval.log
done