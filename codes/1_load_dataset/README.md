# データセットのロード
## ここでは､HuggingFaceのDatasetsライブラリを諸々loadして､一つのjsonlを書き出します
- Datasetsは清掃済みを想定中です
    - クリーニングについては､[このあたり](https://note.com/kan_hatakeyama/n/n331bda7d77c1)を参照
    - datasetsへのラッピング法は[こちら by yamada](https://colab.research.google.com/drive/11rl9Wie22JVIB5bjj3W6bnygfWFlNijW?usp=sharing)
- integrate_dataset.ipynbを実行します｡
- あるいは
~~~
python integrate_dataset.py
~~~

- 実行すると､[data](../../data)フォルダに､全てのテキストを集約した超巨大なjsonlファイル(integrated_text.jsonl)が生成されます