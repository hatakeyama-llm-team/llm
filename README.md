# LLMの開発レポジトリ(作成中)です
## 参考
- セットアップなど
  - https://note.com/kan_hatakeyama/n/nbea55ed4498d
- 標準コード
  - https://github.com/matsuolab/ucllm_nedo_prod


# 環境構築メモ
- 前提として､cuda (nvcc)の11.8が必要｡ドライバは新しくても問題ない
- [cuda toolkit](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=18.04&target_type=runfile_local)や､[複数のcudaを入れる方法](https://qiita.com/takeajioka/items/8737fab5cffbe0118fea)などを参照
~~~
#installの例 (driverは元のまま､cuda toolkitのみ別途入れるようにしました)
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo bash cuda_11.8.0_520.61.05_linux.run

#必要に応じ､パスを通しておく
export PATH="/usr/local/cuda-11.8/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"
~~~
- インストール関係
~~~
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
~~~
- その他
  - wikipedia (en)をdatasetsから使う場合は､apache-beamを使う
~~~
pip install apache-beam==2.54.0
~~~