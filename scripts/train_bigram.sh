NUM_TRAIN_SAMPLES=10000000

OUTPUT_DIR="outputs/bigram_model"
mkdir -p $OUTPUT_DIR

# get dataset names from path
for f in /share/edc/home/alon_albalak/data/pile/test/*; do
    DATASET_NAME=$(basename $f)
    DATASET_NAME=${DATASET_NAME%.jsonl}
    echo $DATASET_NAME

    python3 bigram_model.py \
        --train \
        --evaluate \
        --dataset_name $DATASET_NAME \
        --train_samples $NUM_TRAIN_SAMPLES \
        > ${OUTPUT_DIR}/${DATASET_NAME}.log 2> ${OUTPUT_DIR}/${DATASET_NAME}.err

done