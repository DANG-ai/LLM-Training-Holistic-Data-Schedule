# GENERAL CONFIGS THAT WILL BE USED FOR ALL RUNS
CONFIGS="hds_config/data/pile.yml hds_config/init.yml hds_config/models/1B.yml hds_config/eval_tasks.yml hds_config/sac_1B.yml hds_config/train_data_weights/original_pile.yml"



WANDB_GROUP="acdom_1B"
DATA_SAMPLING_METHOD="sac"
DATA_SAMPLING_WARMUP_STEPS="834"
DATA_SAMPLING_UPDATE_FREQUENCY="1"
MIXED_MINIBATCHES=true
SMOOTHING_FACTOR="0.5"

SEEDS=(42)
# RUN SPECIFIC CONFIGS
for SEED in ${SEEDS[@]}; do
    RUN_NAME="${WANDB_GROUP}_seed${SEED}"
    ARGS="--seed ${SEED} --save outputs/${RUN_NAME} --wandb_group ${WANDB_GROUP} --wandb_run_name seed${SEED} --data_sampling_method ${DATA_SAMPLING_METHOD} --data_sampling_warmup_steps ${DATA_SAMPLING_WARMUP_STEPS} --data_sampling_update_frequency ${DATA_SAMPLING_UPDATE_FREQUENCY} --mixed_minibatches ${MIXED_MINIBATCHES} --data_sampling_smoothing_factor ${SMOOTHING_FACTOR}"
    python3 tools/create_run_specific_config.py ${ARGS}
    RUN_SPECIFIC_CONFIG="hds_config/run_specific/${RUN_NAME}.yml"
    echo "Running with configs: ${CONFIGS} ${RUN_SPECIFIC_CONFIG}"
    python3 deepy.py train.py ${CONFIGS} ${RUN_SPECIFIC_CONFIG} 2>&1 | tee outputs/${RUN_NAME}.log

    wait < <(jobs -p)
done