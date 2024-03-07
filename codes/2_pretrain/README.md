# 事前学習します
## tokenizer
- 設定は[yaml](sentence_piece_config.yaml)
~~~
python 1_train_sentencepiece_tokenizer.py 
~~~
- 実行すると､[tokenizer](../../data/tokenizer/)フォルダにmodelと[vocab](../../data/tokenizer/tokenizer.vocab)ファイルが生成されます
### TODO:
- tokenizerの種類やハイパラの最適化

## pretrain
- Megatron-Deepspeedを入れます(1回のみ実行でOKです)
~~~
bash install_megatron_ds.sh
~~~

- 学習コードを走らせます
  - 以下はone node, one/multi gpu用のcodeです｡ 学習のハイパラもハードコードされています
  - 実行すると､はじめに[data](../../data/) folerに､tokenizeされたデータ(tokenized_text_document.bin/idx)が生成されます｡
  - その後､しばらく待っていると､学習が始まります｡cuda out of memoryの場合は､50行目付近の､global_batch_sizeを減らします
    - 125Mで､global_batch_size=128とすると､A100 (80GB) x2 で57GB x2 程度でした｡

~~~
bash install_megatron_ds.sh
~~~

### TODO
- Wandbとの連携
- 学習途中からのスタート