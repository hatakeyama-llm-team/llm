#Megatron-DeepSpeedのレポジトリをクローン。
git clone https://github.com/hotsuyuki/Megatron-DeepSpeed

# mainブランチではエラーが起きる場合があるため、指定のタグにチェックアウト。
cd Megatron-DeepSpeed/
git fetch origin && git checkout refs/tags/ucllm_nedo_dev_v20240205.1.0

#install
python setup.py install