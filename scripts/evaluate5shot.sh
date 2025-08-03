MODEL_CONFIG=$1
EVAL_CONFIG=$2
STEP=41667
NUM_FEWSHOT=${4:-5}
GPU=${5:-0}

# Can't use perplexity-based evaluation tasks with in-context examples
if [ ${NUM_FEWSHOT} -eq 0 ]; then
    EVAL_TASKS="mmlu"
else
    # if using few-shot, then we can't use wikitext
    EVAL_TASKS="mmlu"
fi

python3 tools/create_eval_config.py --config_path ${MODEL_CONFIG} --num_fewshot ${NUM_FEWSHOT} --iteration ${STEP}

# if not using num_fewshot, then you can just use the following:
if [ ${NUM_FEWSHOT} -eq 0 ]; then
    # EVAL_MODEL_CONFIG is MODEL_CONFIG with .yml replaced by _eval.yml
    EVAL_MODEL_CONFIG=${MODEL_CONFIG%.yml}_eval.yml
else
    EVAL_MODEL_CONFIG=${MODEL_CONFIG%.yml}_eval_${NUM_FEWSHOT}shot.yml
fi

# Get GPU Config
GPU_CONFIG=hds_configs/gpu/gpu${GPU}.yml


# EVAL_CONFIG should be in the configs folder.
python ./deepy.py eval.py ${EVAL_MODEL_CONFIG} ${EVAL_CONFIG} --eval_tasks ${EVAL_TASKS}