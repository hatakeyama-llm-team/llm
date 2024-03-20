# 事前学習します
## 1. tokenizerの学習
- 設定は[yaml](sentence_piece_config.yaml)をいじります｡
  - 主な設定項目
    - input: 学習データ(jsonl)のパス
    - output_dir: 学習したトークナイザーの保存パス
- Wikipedia 200万文書で15 minほどかかりました｡
- 40GBの文章で､5 hr(?)ほど

~~~
python 1_train_sentencepiece_tokenizer.py 
~~~

- 実行すると､[tokenizers](../../model/tokenizers/)フォルダにmodelとvocabファイルが生成されます
  - vocabには､語彙が収録されています｡一度見てみるのがオススメです｡

## 2. tokenize
~~~
bash 2_tokenize.sh
~~~

- tokenizeを実行すると､[data](../../data/) folerに､tokenizeされたデータ(例：tokenized_text_document.bin/idx)が生成されます｡
  - wikipedia記事200万件を処理するのに1000 secほど｡

## 3. pretrain
- 学習の前に､データのtoken数を確認しておきます｡
  - 愚直にカウントしています｡もっと早い方法があるはずです｡
~~~
python count_tokens.py
~~~
- 算出されたtoken数を､[config](config.yaml)に反映させます｡
  - train_tokensを変更します｡
  - こうすると､1epoch分だけ学習されるようになります｡
- 必要に応じ､同ファイルのパラメータを変更します｡
  - モデルパラメータもこのファイル内で色々といじれます
  - global_batch_sizeを小さくすると､必要なVRAMを削減できます｡
    - VRAMの目安
    - 125Mで､global_batch_size=128とすると､A100 (80GB) x2 で57GB x2 程度
    - 300Mでは72 (zero stage 1)で75 GB x2 
- 一番初めの実行はcompile?が入るようで､時間がかかります｡

- wandbを使うように設定を変更します。
  - project名は、[config](./original_codes/ds_config_gpt_TEMPLATE.json)のwandb-projectから変更してください。
~~~
cp original_codes/ds_config_gpt_TEMPLATE.json Megatron-DeepSpeed/examples_deepspeed/rebase/ds_config_gpt_TEMPLATE.json
~~~

- 学習の実行
~~~
bash 3_train_node1.sh
~~~


### エラー対策
- 事前学習スクリプトが "> compiling and loading fused kernels ..." というところでスタック
  - DeepSpeedのfused_kernelのbuildをやり直すため、一旦、削除する
~~~
rm -rf Megatron-DeepSpeed/megatron/fused_kernels/build/
~~~
- -9でプロセスがkill
  - メモリ(RAM)不足なので、config.yamlのtrain_samplesを小さくする
- cuda out of memory
  - GPUメモリ(VRAM)不足なので、config.yamlのglobal_batch_sizeを小さくする  

## HuggingFace形式へのモデル変換
- 無事に学習がおわると､[こちら](../../models/pretrain/gpt/checkpoint/)フォルダ内にモデルデータ群が生成されます｡
  - この中から､最新のcheckpointなどを選びます
  - [converrt_config](./convert_config.yaml)を開き､設定します｡
    - input_model_dir: checkpointのフォルダ
    - output_tokenizer_and_model_dir: huggingfaceのレポにuploadする際の名前
~~~
bash 4_convert_to_HF.sh
~~~

## (遊び): 作ったモデルを動かしてみる
- [こちらのnotebook](./5_play_with_model.ipynb)

## モデルのアップロード
- HuggingFaceのレポジトリにuploadします｡
  - モデル名は[config](./convert_config.yaml)のhugging_face_nameで指定します｡
~~~
6_upload.py
~~~

### TODO
- Wandbとの連携
