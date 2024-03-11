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

- セットアップスクリプト  (詳細は[こちら](https://note.com/kan_hatakeyama/n/nbea55ed4498d))
~~~
cd codes
bash setup.sh
~~~

# 学習法
## [1_load_dataset](./codes/1_load_dataset/)
- datasetsライブラリをもとに、データを読み込んで行きます
- Branch-Train-Merge/カリキュラム学習的な考えを想定したシステム設計なので、どのデータ配分などもここで決めます
  
## [2_pretrain](./codes/2_pretrain/)
- トークナイザーの学習、トークナイズ、事前学習、HuggingFace modelへの変換を行います。

## ファインチューニング、評価
- 今はあまり必要ないのと、そこまで実装難易度が高くないので、今後の適当なタイミングで作ります。

# 全自動での学習: 作業中
~~~
cd codes
bash auto.sh
~~~