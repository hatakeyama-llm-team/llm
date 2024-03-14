# 事前学習します
## 1. tokenizerの学習
- 設定は[yaml](sentence_piece_config.yaml)をいじります｡
  - 主な設定項目
    - input: 学習データ(jsonl)のパス
    - output_dir: 学習したトークナイザーの保存パス
- Wikipedia 200万文書で15 minほどかかりました｡

~~~
python 1_train_sentencepiece_tokenizer.py 
~~~

- 実行すると､[tokenizers](../../model/tokenizers/)フォルダにmodelとvocabファイルが生成されます
  - vocabには､語彙が収録されています｡一度見てみるのがオススメです｡

## 2. tokenize
~~~
bash 2_tokenize.sh
~~~

## pretrain
- 学習コードを走らせます
- tokenizeを実行すると､[data](../../data/) folerに､tokenizeされたデータ(例：tokenized_text_document.bin/idx)が生成されます｡
  - wikipedia記事200万件を処理するのに1000 secほど｡
~~~
bash 2_tokenize.sh

~~~
- pretrainを実行すると、学習が始まります｡cuda out of memoryの場合は､global_batch_sizeを減らします
  - 125Mで､global_batch_size=128とすると､A100 (80GB) x2 で57GB x2 程度でした｡
  - 300Mでは72 (zero stage 1)で75 GB x2 でした


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