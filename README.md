# LLMの開発レポジトリ(作成中)です
## 参考
- セットアップなど
  - https://note.com/kan_hatakeyama/n/nbea55ed4498d
- 標準コード
  - https://github.com/matsuolab/ucllm_nedo_prod

# Dockerで環境構築する場合(ファインチューニングまで動作確認済み)
- Dockefile made by ssone
- 数十分はかかります｡
~~~
git clone https://github.com/hatakeyama-llm-team/llm.git
sudo docker build -t llm .
~~~

- 実行
~~~
#sudo docker run --gpus all --rm -it -p 8899:8888 -v .:/home/llm llm bash

#1回目の実行
sudo docker run --gpus all --shm-size='1gb' -it -p 8899:8888 -v .:/home/llm llm bash

#2回目以降
sudo docker start -i ...


sudo chmod -R 777 llm
cd llm/
conda activate scr

#初回起動時は以下のsetup scriptを実行します｡
bash docker_setup.sh

#huggingfaceなどもログインします
huggingface-cli login
wandb login 

~~~



# 直接､環境構築する場合
- cuda (nvcc)の11.8が必要｡ドライバは新しくても問題ない
- [cuda toolkit](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=18.04&target_type=runfile_local)や､[複数のcudaを入れる方法](https://qiita.com/takeajioka/items/8737fab5cffbe0118fea)などを参照
~~~
#installの例 (driverは元のまま､cuda toolkitのみ別途入れればOK)
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo bash cuda_11.8.0_520.61.05_linux.run

#必要に応じ､パスを通しておく
export PATH="/usr/local/cuda-11.8/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"
~~~

- セットアップスクリプト  (詳細は[こちら](https://note.com/kan_hatakeyama/n/nbea55ed4498d))
~~~
cd codes
bash setup.sh
~~~

# 学習法
## [1 データセット構築](./codes/1_load_dataset/)
- datasetsライブラリをもとに、データを読み込んで行きます
- Branch-Train-Merge/カリキュラム学習的な考えを想定したシステム設計なので、どのデータ配分などもここで決めます
  
## [2 事前学習](./codes/2_pretrain/)
- トークナイザーの学習、トークナイズ、事前学習、HuggingFace modelへの変換を行います。

## [3 ファインチューニング](./codes/3_finetune/)
- ファインチューニングします｡

## [4 評価](./codes/4_eval/)
- 構築したモデルを評価します｡

# 全自動での学習: 作業中
~~~
cd codes
bash auto.sh
~~~