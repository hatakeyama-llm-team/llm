#動作未確認です

# conda
conda create -n scr python=3.11 -y
conda activate scr

#torch類のinstall
pip install torch==2.0.1+cu118 torchaudio==2.0.2+cu118 torchvision==0.15.2+cu118 --find-links https://download.pytorch.org/whl/torch_stable.html

#requirements: accelerate, transformers, sentencepieceのような言語モデル系ライブラリ
pip install -r requirements.txt

#deepspeed
pip install deepspeed-kernels

#deepspeedのbuild用
sudo apt-get install libaio-dev -y

# deepspeedのbuild (15分くらいかかる｡****)
DS_BUILD_OPS=1 DS_BUILD_EVOFORMER_ATTN=0 DS_BUILD_SPARSE_ATTN=0 pip install deepspeed==0.12.4

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
cd ../
pip uninstall ninja -y && pip install ninja==1.11.1
pip install flash-attn==2.5.0 --no-build-isolation

#sft
# mainブランチではエラーが起きる場合があるため、指定のタグにチェックアウト。
git clone https://github.com/hotsuyuki/llm-jp-sft
cd llm-jp-sft
git fetch origin
git checkout refs/tags/ucllm_nedo_dev_v20240208.1.0

#megatron-deepspeed
cd 2_pretrain
#Megatron-DeepSpeedのレポジトリをクローン。
git clone https://github.com/hotsuyuki/Megatron-DeepSpeed

# mainブランチではエラーが起きる場合があるため、指定のタグにチェックアウト。
cd Megatron-DeepSpeed/
git fetch origin && git checkout refs/tags/ucllm_nedo_dev_v20240205.1.0

#install
python setup.py install

cd ../

#wikipediaをdatasetsから使う場合
pip install apache-beam==2.54.0

#yaml
pip install yq==3.2.3
sudo apt-get install jq -y