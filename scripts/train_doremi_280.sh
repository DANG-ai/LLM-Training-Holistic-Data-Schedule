# GENERAL CONFIGS THAT WILL BE USED FOR ALL RUNS
CONFIGS="alon_configs/data/pile.yml alon_configs/init.yml alon_configs/train_data_weights/doremi_280.yml"
# Doesn't include alon_configs/eval_tasks.yml or  alon_configs/parallelism.yml

# RUN SPECIFIC CONFIGS
CONFIGS_160M="alon_configs/models/160m.yml alon_configs/run_specific/160m_doremi_280.yml"
CONFIGS_410M="alon_configs/models/410m.yml alon_configs/run_specific/410m_doremi_280.yml"
CONFIGS_1B="alon_configs/models/1B.yml alon_configs/run_specific/1B_doremi_280.yml"

echo "Running with configs: ${CONFIGS} ${CONFIGS_160M}"
RUN_NAME="160m_doremi_280"
python3 deepy.py train.py ${CONFIGS} ${CONFIGS_160M} 2>&1 | tee outputs/${RUN_NAME}.log

echo "Running with configs: ${CONFIGS} ${CONFIGS_410M}"
RUN_NAME="410m_doremi_280"
python3 deepy.py train.py ${CONFIGS} ${CONFIGS_410M} 2>&1 | tee outputs/${RUN_NAME}.log

echo "Running with configs: ${CONFIGS} ${CONFIGS_1B}"
RUN_NAME="1B_doremi_280"
python3 deepy.py train.py ${CONFIGS} ${CONFIGS_1B} 2>&1 | tee outputs/${RUN_NAME}.log