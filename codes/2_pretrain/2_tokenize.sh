# tokenize

#data_path=$(yq -r '.data_path' config.yaml)
output_prefix=$(yq -r '.output_prefix' config.yaml)
megatron_deepspeed_dir=$(yq -r '.megatron_deepspeed_dir' config.yaml)
input_jsonl=$(yq -r '.input_jsonl' config.yaml)
input_tokenizer_file=$(yq -r '.input_tokenizer_file' config.yaml)
echo "tokenizer-model: ${input_tokenizer_file}"

python ${megatron_deepspeed_dir}/tools/preprocess_data.py \
    --tokenizer-type SentencePieceTokenizer \
    --tokenizer-model ${input_tokenizer_file} \
    --input  ${input_jsonl} \
    --output-prefix ${output_prefix} \
    --dataset-impl mmap \
    --workers 64 \
    --append-eod
echo ""