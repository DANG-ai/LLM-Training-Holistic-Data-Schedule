# GENERAL CONFIGS THAT WILL BE USED FOR ALL RUNS
CONFIGS="alon_configs/data/pile.yml alon_configs/models/410m.yml alon_configs/init.yml alon_configs/optimizer.yml"
# Doesn't include alon_configs/eval_tasks.yml or  alon_configs/parallelism.yml

# RUN SPECIFIC CONFIGS
ORIGINAL_WEIGHT_CONFIGS="alon_configs/train_data_weights/original_pile.yml alon_configs/run_specific/410m_original.yml"
DOREMI_280_CONFIGS="alon_configs/train_data_weights/doremi_280.yml alon_configs/run_specific/410m_doremi_280.yml"
DOREMI_1B_CONFIGS="alon_configs/train_data_weights/doremi_1B.yml alon_configs/run_specific/410m_doremi_1B.yml"

echo "Running with configs: ${CONFIGS} ${ORIGINAL_WEIGHT_CONFIGS}"
RUN_NAME="410m_original"
python3 deepy.py train.py ${CONFIGS} ${ORIGINAL_WEIGHT_CONFIGS} 2>&1 | tee outputs/${RUN_NAME}.log

echo "Running with configs: ${CONFIGS} ${DOREMI_280_CONFIGS}"
RUN_NAME="410m_doremi_280"
python3 deepy.py train.py ${CONFIGS} ${DOREMI_280_CONFIGS} 2>&1 | tee outputs/${RUN_NAME}.log

echo "Running with configs: ${CONFIGS} ${DOREMI_1B_CONFIGS}"
RUN_NAME="410m_doremi_1B"
python3 deepy.py train.py ${CONFIGS} ${DOREMI_1B_CONFIGS} 2>&1 | tee outputs/${RUN_NAME}.log