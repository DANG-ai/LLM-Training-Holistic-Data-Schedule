# GENERAL CONFIGS THAT WILL BE USED FOR ALL RUNS
CONFIGS="hds_config/data/pile.yml hds_config/init.yml hds_config/models/1B.yml hds_config/eval_tasks.yml hds_config/sac_1B.yml hds_config/train_data_weights/original_pile.yml"



WANDB_GROUP="acdom_410m_1B"
DATA_SAMPLING_METHOD="sac"
DATA_SAMPLING_WARMUP_STEPS="834"
DATA_SAMPLING_UPDATE_FREQUENCY="1"
MIXED_MINIBATCHES=true
SMOOTHING_FACTOR="0.5"


SEEDS=(42)
# RUN SPECIFIC CONFIGS
for SEED in ${SEEDS[@]}; do
    RUN_NAME="${WANDB_GROUP}_seed${SEED}"
    # evaluate
    bash scripts/evaluate_he.sh outputs/${RUN_NAME}/global_step41667/configs/${RUN_NAME}.yml hds_config/models/eval_1B_1gpu.yml 2>&1 | tee outputs/${RUN_NAME}_eval.log &

done