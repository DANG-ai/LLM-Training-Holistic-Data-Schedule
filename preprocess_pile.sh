domains=('Pile-CC' 'ArXiv' 'BookCorpus2' 'PubMed_Abstracts' 'PubMed_Central' 'Ubuntu_IRC' 'Books3' 'DM_Mathematics' 'Enron_Emails' 'EuroParl' 'FreeLaw' 'Gutenberg_(PG-19)' 'HackerNews' 'NIH_ExPorter' 'OpenSubtitles' 'PhilPapers' 'YoutubeSubtitles' 'Github' 'OpenWebText2' 'StackExchange' 'USPTO_Backgrounds' 'Wikipedia_(en)')

INPUT_BASE_DIR=$1
OUTPUT_BASE_DIR=$2

for split in train test;
do
    SPLIT_PATH="${INPUT_BASE_DIR}/${split}/"
    for DATASET_NAME in "${domains[@]}";
    do
        DATASET_PATH="${SPLIT_PATH}${DATASET_NAME}.jsonl"
        OUTPUT_DIR=${OUTPUT_BASE_DIR}$/$split/${DATASET_NAME}
        echo "path: dataset path: ${DATASET_PATH}"
        echo "name: dataset name: ${DATASET_NAME}"
        echo "outputting to: ${OUTPUT_DIR}"

        mkdir -p ${OUTPUT_DIR}

        python tools/preprocess_data.py \
            --input $DATASET_PATH \
            --output-prefix ${OUTPUT_DIR}/${DATASET_NAME} \
            --vocab-file /home/cetc15/code/online-data-mixing-pythia-sac/tokenizer_config/20B_tokenizer.json \
            --dataset-impl mmap \
            --tokenizer-type HFTokenizer \
            --workers 16 \
            --append-eod 2>&1 | tee ${OUTPUT_DIR}.log
    done
done
