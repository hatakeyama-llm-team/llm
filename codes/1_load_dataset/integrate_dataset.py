# %%
# 種々のdatasetをintagrateします

# %%
from src.loaders import *
import random
from src.RecordDistributor import RecordDistributor
import json
import os
from tqdm import tqdm
import yaml
import os

def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

make_dir("../../data")
make_dir("../../data/text")

with open('config.yaml', 'r') as file:
    conf = yaml.safe_load(file)
output_path = conf["output_path"]
max_records = conf["max_records"]
print(conf)


"""
loader: datasetsのloaderを入れてます｡ 
n_records: 最大の学習件数
stage_ratio: 各ステージで､どの程度､データを食わせるかを決めます｡

例:
wiki(ja): [1,1,9]
wiki(en): [1,9,1]
の場合､
データセットを3stageに分けます｡

1st stageでは､ wiki(ja)の10%､wiki(en)の10%を混ぜて学習させます
2nd stageでは､ wiki(ja)の10%､wiki(en)の90%を混ぜて学習させます
3rd stageでは､ wiki(ja)の90%､wiki(en)の10%を混ぜて学習させます

このようなステージ分けをすることで､一種のカリキュラム学習を行うことが出来ます
"""

dataset_dict = {
    "wiki(ja)": {
        "loader": wiki_ja_loader, #日本語版のwikipediaのloaderを使います｡
        "n_records": max_records, #最大件数
        "stage_ratio": [1, 1, 1,8],  # 各ステージでのデータ配分
    },
    "wiki(en)": {
        "loader": wiki_en_loader,
        "n_records": max_records,
        "stage_ratio": [1, 8, 1,1],
    },
    "mc4(ja)": {
        "loader": mc4_ja_part_loader,
        "n_records": max_records,
        "stage_ratio": [1, 1, 8,1],  # 各ステージでのデータ配分
    },
}

# %%
distributor = RecordDistributor(dataset_dict)
distributor.load_datasets()

# %%
distributor.dataset_dict, distributor.n_records_per_stage, distributor.n_records_per_stage

# %%
distributor.write_jsonl(output_path, overwrite=conf["overwrite"])

# %%
