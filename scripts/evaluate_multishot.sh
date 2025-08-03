RUN_NAME=$1
MODEL_CONFIG_EVAL=$2
STEP=$3

# MODEL_CONFIG_EVAL is the file name in the alon_configs/models/ directory
# e.g. alon_configs/models/eval_1B_1gpu.yml
# OR alon_configs/models/eval_3B_seqlen2048_1gpu.yml

# evaluate 0-shot
bash scripts/evaluate.sh outputs/${RUN_NAME}/global_step${STEP}/configs/${RUN_NAME}.yml alon_configs/models/${MODEL_CONFIG_EVAL}.yml ${STEP} 0 2>&1 | tee outputs/${RUN_NAME}_${STEP}_eval.log &
# evaluate 1-shot through 5-shot
for i in {1..5}; do
    bash scripts/evaluate.sh outputs/${RUN_NAME}/global_step${STEP}/configs/${RUN_NAME}.yml alon_configs/models/${MODEL_CONFIG_EVAL}.yml ${STEP} ${i} ${i} 2>&1 | tee outputs/${RUN_NAME}_${STEP}_${i}shot_eval.log &
done
wait < <(jobs -p)