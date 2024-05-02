#!/bin/bash

set -e
echo ""

# Stores the directory paths as variables.
megatron_deepspeed_dir="Megatron-DeepSpeed"
echo "megatron_deepspeed_dir = ${megatron_deepspeed_dir}"
echo ""

# Initializes the arguments.
input_tokenizer_file=""
input_model_dir=""
output_tokenizer_and_model_dir=""
temp_model_dir=""
model_name=""
target_tp=1
target_pp=1

# Parses the arguments.
while [[ ${#} -gt 0 ]]; do
    case ${1} in
        # Shifts twice for option that takes an argument.
        --input_tokenizer_file) input_tokenizer_file=${2}; shift ;;
        --input_model_dir) input_model_dir=${2}; shift ;;
        --output_tokenizer_and_model_dir) output_tokenizer_and_model_dir=${2}; shift ;;
        --temp_model_dir) temp_model_dir=${2}; shift ;;
        --model_name) model_name=${2}; shift ;;
        --target_tp) target_tp=${2}; shift ;;
        --target_pp) target_pp=${2}; shift ;;
        *) echo "Unknown parameter passed: ${1}"; exit 1 ;;
    esac
    # Shifts once per loop to move to the next key/value.
    shift
done

# Checks the required arguments.
if [[ -z ${input_tokenizer_file} ]] || [[ -z ${input_model_dir} ]] || [[ -z ${output_tokenizer_and_model_dir} ]] || [[ -z ${model_name} ]]; then
    echo "Error: Missing required arguments."
    echo "Usage: ${0} --input_tokenizer_file <input_tokenizer_file> --input_model_dir <input_model_dir> --output_tokenizer_and_model_dir <output_tokenizer_and_model_dir> --moel_name <model_name>"
    exit 1
fi

# Prints the arguments.
echo "input_tokenizer_file = ${input_tokenizer_file}"
echo "input_model_dir = ${input_model_dir}"
echo "output_tokenizer_and_model_dir = ${output_tokenizer_and_model_dir}"
echo "temp_model_dir = ${temp_model_dir}"
echo "model_name = ${model_name}"
echo "target_tp = ${target_tp}"
echo "target_pp = ${target_pp}"
echo ""

mkdir -p ${output_tokenizer_and_model_dir}
mkdir -p ${temp_model_dir}

# Converts the tokenizer from SentencePiece format to HuggingFace Transformers format.
#python convert_tokenizer_from_sentencepiece_to_huggingface_transformers.py \
#    --input_tokenizer_file ${input_tokenizer_file} \
#    --output_tokenizer_dir ${output_tokenizer_and_model_dir} \
#    --tokenizer_type LlamaTokenizer

# Converts the pretrained model from Megatron-DeepSpeed format to HuggingFace Transformers format.
python ${megatron_deepspeed_dir}/tools/convert_checkpoint/deepspeed_to_megatron.py \
    --input_folder ${input_model_dir} \
    --output_folder ${temp_model_dir} \
    --target_pp ${target_pp} \
    --target_tp ${target_tp}

python llama_checkpoint_conversion.py \
    --convert_checkpoint_from_megatron_to_transformers \
    --save_path ${output_tokenizer_and_model_dir} \
    --print-checkpoint-structure \
    --load_path ${temp_model_dir}   \
    --model_name ${model_name}
   
echo ""
echo "Finished to convert the tokenizer and the pretrained model to HuggingFace Transformers format."
echo ""
