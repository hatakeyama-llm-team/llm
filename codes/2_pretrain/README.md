# 事前学習します
## tokenizer
- 設定は[yaml](sentence_piece_config.yaml)
- Wikipedia 200万文書で15 minほどかかりました｡

~~~
python 1_train_sentencepiece_tokenizer.py 
~~~
- 実行すると､[tokenizer](../../data/tokenizer/)フォルダにmodelと[vocab](../../data/tokenizer/tokenizer.vocab)ファイルが生成されます
### TODO:
- tokenizerの種類やハイパラの最適化

## pretrain
- 学習コードを走らせます
  - 以下はone node, one/multi gpu用のcodeです｡ 学習のハイパラもハードコードされています
  - 実行すると､はじめに[data](../../data/) folerに､tokenizeされたデータ(tokenized_text_document.bin/idx)が生成されます｡
    - データベースを更新した場合は､これらのファイルを削除すること
    - wikipedia記事200万件を処理するのに1000 secほど｡
  - tokenize後､しばらく待っていると､学習が始まります｡cuda out of memoryの場合は､50行目付近の､global_batch_sizeを減らします
    - 125Mで､global_batch_size=128とすると､A100 (80GB) x2 で57GB x2 程度でした｡

~~~
bash 2_tokenize.sh
bash 3_train_node1.sh
~~~

## モデル変換
- [converrt_configを開く](./convert_config.yaml)
- input_model_dirに､最新のモデルのcheckpointのディレクトリを指定する
  - -[ここから探す](../../models/gpt/checkpoint/)
~~~
bash 4_convert_to_HF.sh
~~~

## (遊び): 作ったモデルを動かしてみる
- [こちらのnotebook](./5_play_with_model.ipynb)

### TODO
- Wandbとの連携