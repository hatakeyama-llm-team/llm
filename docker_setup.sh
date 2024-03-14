
#Megatron-DeepSpeedのレポジトリをクローン。
cd codes/2_pretrain
git clone https://github.com/hotsuyuki/Megatron-DeepSpeed

# mainブランチではエラーが起きる場合があるため、指定のタグにチェックアウト。
cd Megatron-DeepSpeed/
git fetch origin && git checkout refs/tags/ucllm_nedo_dev_v20240205.1.0

#install
python setup.py install

cd ../

#yq
pip install yq==3.2.3
sudo apt-get install jq -y

#apex
git clone https://github.com/NVIDIA/apex
cd apex
# mainブランチではエラーが起きる場合があるため、指定のタグにチェックアウト。
git fetch origin && git checkout refs/tags/23.08

#install
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

# apex_C.cpython-311-x86_64-linux-gnu.soが作成されていることを確認。
find build/lib.linux-x86_64-cpython-311/ -name apex_C.cpython-311-x86_64-linux-gnu.so

#flash atten
pip uninstall ninja -y && pip install ninja==1.11.1
pip install flash-attn==2.5.0 --no-build-isolation

git config --global --add safe.directory /home/llm


cd ../
#データ学習時に､読み込みがshuffleされるコードを無効化 (BTMは学習順序が大切なので｡)
cp codes/2_pretrain/original_codes/gpt_dataset.py codes/2_pretrain/Megatron-DeepSpeed/megatron/data/gpt_dataset.py

cd codes/3_finetune
git config --global --add safe.directory /home/llm/codes/3_finetune/llm-jp-sft
# mainブランチではエラーが起きる場合があるため、指定のタグにチェックアウト。
git clone https://github.com/hotsuyuki/llm-jp-sft
cd llm-jp-sft
git fetch origin
git checkout refs/tags/ucllm_nedo_dev_v20240208.1.0



