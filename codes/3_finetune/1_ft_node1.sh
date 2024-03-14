#!/bin/bash

set -e
echo ""

# Stores the directory paths as variables.
#ucllm_nedo_dev_train_dir="${HOME}/ucllm_nedo_dev/train"

megatron_deepspeed_dir=$(yq -r '.megatron_deepspeed_dir' ../2_pretrain/config.yaml)

echo "megatron_deepspeed_dir = ${megatron_deepspeed_dir}"
echo ""

# Initializes the arguments.
input_model_name_or_path=$(yq -r '.huggingface_input_model' config.yaml)
output_tokenizer_and_model_dir=$(yq -r '.output_tokenizer_and_model_dir' config.yaml)

# Prints the arguments.
echo "input_model_name_or_path = ${input_model_name_or_path}"
echo "output_tokenizer_and_model_dir = ${output_tokenizer_and_model_dir}"
echo ""

mkdir -p ${output_tokenizer_and_model_dir}


dataset_file=$(yq -r '.dataset_path' config.yaml)
echo ""

# Logging.
log_path="${output_tokenizer_and_model_dir}/log"
mkdir -p ${log_path}
host="${HOSTNAME}"
current_time=$(date "+%Y.%m.%d_%H.%M.%S")

# Finetunes the pretrained model.
python ./llm-jp-sft/train.py \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --bf16 \
    --data_files ${dataset_file} \
    --model_name_or_path ${input_model_name_or_path} \
    --output_dir ${output_tokenizer_and_model_dir} \
    --instruction_template "### 指示:" \
    --response_template "### 応答:" \
    2>&1 | tee ${log_path}/${host}_${current_time}.log

#    --instruction_template "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。\n\n### 指示:\n" \
#    --response_template "\n\n### 応答:\n" \
#    --instruction_template "### Human:" \
#    --response_template "### Assistant:" \
echo ""
echo "Finished to finetune the pretrained model."
echo ""
