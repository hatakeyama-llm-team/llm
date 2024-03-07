# tokenize

data_path=$(yq -r '.data_path' config.yaml)
megatron_deepspeed_dir=$(yq -r '.megatron_deepspeed_dir' config.yaml)

python ${megatron_deepspeed_dir}/tools/preprocess_data.py \
    --tokenizer-type SentencePieceTokenizer \
    --tokenizer-model ${input_tokenizer_file} \
    --input ../../data/integrated_text.jsonl \
    --output-prefix ../../data/tokenized \
    --dataset-impl mmap \
    --workers 64 \
    --append-eod
echo ""