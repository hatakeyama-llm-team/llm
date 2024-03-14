#!/bin/bash

set -e
echo "begin converting.."

# Stores the directory paths as variables.
megatron_deepspeed_dir=$(yq -r '.megatron_deepspeed_dir' config.yaml)
echo "megatron_deepspeed_dir = ${megatron_deepspeed_dir}"
echo ""


input_tokenizer_file=$(yq -r '.input_tokenizer_file' config.yaml)
input_model_dir=$(yq -r '.input_model_dir' convert_config.yaml)
output_tokenizer_and_model_dir=$(yq -r '.output_tokenizer_and_model_dir' convert_config.yaml)


# Prints the arguments.
echo "input_tokenizer_file = ${input_tokenizer_file}"
echo "input_model_dir = ${input_model_dir}"
echo "output_tokenizer_and_model_dir = ${output_tokenizer_and_model_dir}"
echo ""

mkdir -p ${output_tokenizer_and_model_dir}

# Converts the tokenizer from SentencePiece format to HuggingFace Transformers format.
python convert_tokenizer_from_sentencepiece_to_huggingface_transformers.py \
    --input_tokenizer_file ${input_tokenizer_file} \
    --output_tokenizer_dir ${output_tokenizer_and_model_dir}

# Converts the pretrained model from Megatron-DeepSpeed format to HuggingFace Transformers format.
python ${megatron_deepspeed_dir}/tools/convert_checkpoint/deepspeed_to_transformers.py \
    --input_folder ${input_model_dir} \
    --output_folder ${output_tokenizer_and_model_dir}

echo ""
echo "Finished to convert the tokenizer and the pretrained model to HuggingFace Transformers format."
echo ""
