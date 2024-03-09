# %%
# 種々のdatasetをintagrateします

# %%
from datasets import load_dataset
import json
import os
from tqdm import tqdm
import yaml

with open('config.yaml', 'r') as file:
    conf = yaml.safe_load(file)
output_path = conf["output_path"]
max_records = conf["max_records"]
print(conf)

# %%
# ここでは例として､日英のwikipediaを読み込んでみます

# TODO: よく考えると､これだとRAMが足りなくなるかも?
dataset_list = [
    load_dataset("wikipedia", "20220301.en", split="train").shuffle(),  # 英語
    load_dataset("hpprc/wikipedia-20240101", split="train").shuffle(),  # 日本語
]


# %%
# 各datasetをmergeして､一つのjsonlにまとめます｡
# ファイルは output_pathに生成されます
# 各datasetは予めクリーニング, dedupされている必要があります｡
# TODO: BTMのため､ジャンル別にデータを並べ替えたい


overwrite = conf["overwrite"]
if overwrite:
    with open(output_path, "a") as f:
        for dataset in dataset_list:
            print(dataset)
            cnt = 0
            for data in tqdm(dataset):
                out_text = json.dumps(
                    {"text": data["text"]}, ensure_ascii=False)
                # jsonlで書き出し

                f.write(out_text+"\n")

                cnt += 1
                if cnt > max_records:
                    break

    # TODO: dataフォルダ内のindex fileを削除
# %%
