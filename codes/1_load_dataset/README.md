# データセットのロード
## ここでは､HuggingFaceのDatasetsライブラリを諸々loadして､一つのjsonlを書き出します
- Datasetはあらかじめクリーニングされたものを用います｡
    - クリーニングについては､[こちらの記事](https://note.com/kan_hatakeyama/n/n331bda7d77c1)などを参照
    - 独自構築したコーパスのdatasets ライブラリへのラッピング法は[こちら by yamada](https://colab.research.google.com/drive/11rl9Wie22JVIB5bjj3W6bnygfWFlNijW?usp=sharing)

- 用いるDatasetは､dataset_dictに記入していきます｡

## 以下のコマンドを実行します
- 設定は[こちら](./config.yaml)
    - データの出力先
- どのデータを用いるかについては､[実行コード](./integrate_dataset.py)を直接いじって作業します｡
~~~
python integrate_dataset.py
~~~

- 実行すると､[data](../../data/text)フォルダに､全てのテキストを集約した超巨大なjsonlファイル(integrated_text.jsonl)が生成されます