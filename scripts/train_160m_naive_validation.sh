# GENERAL CONFIGS THAT WILL BE USED FOR ALL RUNS
CONFIGS="alon_configs/data/pile.yml alon_configs/init.yml alon_configs/models/160m.yml alon_configs/eval_tasks.yml"
# Doesn't include alon_configs/parallelism.yml

WANDB_GROUP="160m_ods_naive_validation_10"
DATA_SAMPLING_METHOD="naive_validation"
DATA_SAMPLING_WARMUP_STEPS="2000"
DATA_SAMPLING_UPDATE_FREQUENCY="10"
MIXED_BATCHES=true
VALIDATION_BASED_REWARD=true

SEEDS=(1234 42 100 222)

# RUN SPECIFIC CONFIGS
for SEED in ${SEEDS[@]}; do
    RUN_NAME="${WANDB_GROUP}_seed${SEED}"
    ARGS="--seed ${SEED} --save outputs/${RUN_NAME} --wandb_group ${WANDB_GROUP} --wandb_run_name seed${SEED} --data_sampling_method ${DATA_SAMPLING_METHOD} --data_sampling_warmup_steps ${DATA_SAMPLING_WARMUP_STEPS} --data_sampling_update_frequency ${DATA_SAMPLING_UPDATE_FREQUENCY} --mixed_batches ${MIXED_BATCHES} --validation_based_reward ${VALIDATION_BASED_REWARD}"
    python3 tools/create_run_specific_config.py ${ARGS}
    RUN_SPECIFIC_CONFIG="alon_configs/run_specific/${RUN_NAME}.yml"
    echo "Running with configs: ${CONFIGS} ${RUN_SPECIFIC_CONFIG}"
    python3 deepy.py train.py ${CONFIGS} ${RUN_SPECIFIC_CONFIG} 2>&1 | tee outputs/${RUN_NAME}.log

    # evaluate
    bash scripts/evaluate.sh outputs/${RUN_NAME}/global_step100000/configs/${RUN_NAME}.yml alon_configs/models/eval_160m_1gpu.yml
done