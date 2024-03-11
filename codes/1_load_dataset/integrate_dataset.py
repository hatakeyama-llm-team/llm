# %%
# 種々のdatasetをintagrateします

# %%
from src.loaders import wiki_en_loader, wiki_ja_loader
import random
from src.RecordDistributor import RecordDistributor
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

# %%
dataset_dict = {
    "wiki(ja)": {
        "loader": wiki_ja_loader,
        "n_records": max_records,
        "stage_ratio": [1, 1, 9],  # 各ステージでのデータ配分
    },
    "wiki(en)": {
        "loader": wiki_en_loader,
        "n_records": max_records,
        "stage_ratio": [1, 9, 1],
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
