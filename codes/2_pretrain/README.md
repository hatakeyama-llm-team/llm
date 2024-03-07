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