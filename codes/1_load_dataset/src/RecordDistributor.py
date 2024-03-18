import os
import numpy as np
import random
import json
from tqdm import tqdm

import pandas as pd




class RecordDistributor:
    def __init__(self, dataset_dict, batch_size=1000) -> None:
        self.dataset_dict = dataset_dict
        self.batch_size = batch_size

        update_records_per_stage(self.dataset_dict)
        self.total_records = get_total_records(self.dataset_dict)
        self.update_n_records_per_stage()
        self.update_call_frequncy()

        #集計結果をdataframeに纏める
        self.record_df,self.percent_df=dict_to_df(self.dataset_dict)

    def load_datasets(self):
        for name, dataset_info in self.dataset_dict.items():
            print(f"loading {name}")
            dataset_info["dataset"] = dataset_info["loader"]()
        self.init_iterators()

    def init_iterators(self):
        for name, dataset_info in self.dataset_dict.items():
            dataset_info["dataset_iterator"] = iter(dataset_info["dataset"])

    def update_n_records_per_stage(self):
        # get total number of stages
        for dataset_info in self.dataset_dict.values():
            self.n_stages = len(dataset_info["records_per_stage"])
            break

        self.n_records_per_stage = np.zeros(self.n_stages)
        self.max_data_records_per_stage = np.zeros(self.n_stages)

        for dataset_info in self.dataset_dict.values():
            for i, records in enumerate(dataset_info["records_per_stage"]):
                self.n_records_per_stage[i] += records
                self.max_data_records_per_stage[i] = max(
                    self.max_data_records_per_stage[i], records)

    def update_call_frequncy(self):
        for dataset_info in self.dataset_dict.values():
            frequency_list = []
            for stage in range(self.n_stages):
                frequency = dataset_info["records_per_stage"][stage] / \
                    self.max_data_records_per_stage[stage]
                frequency_list.append(frequency)

            dataset_info["call_frequency"] = frequency_list

    def write_jsonl(self, output_path, overwrite=True):
        self.init_iterators()
        if overwrite:
            with open(output_path, "w") as f:
                f.write("")
        else:
            if os.path.exists(output_path):
                print("file already exists")
                raise FileExistsError

        # write files
        for stage in range(self.n_stages):
            print(f"writing stage {stage}")
            text_list = []
            for cnt in tqdm(range(int(self.max_data_records_per_stage[stage]))):
                batch_cnt = cnt % self.batch_size

                # 各データセットの出現頻度に応じて、データを吐き出していく
                for dataset_info in self.dataset_dict.values():
                    frequency = dataset_info["call_frequency"][stage]

                    # frequency
                    if frequency*self.batch_size > batch_cnt:
                        try:
                            text = next(dataset_info["dataset_iterator"])
                            text_list.append(text["text"])
                        except Exception as e:
                            print(e)

                # バッチにデータが溜まったら、シャッフルして書き出す
                if batch_cnt == self.batch_size-1:
                    random.shuffle(text_list)

                    with open(output_path, "a") as f:
                        for text in text_list:
                            out_text = json.dumps(
                                {"text": text}, ensure_ascii=False)
                            f.write(out_text+"\n")
                    text_list = []

def update_records_per_stage(dataset_dict):
    for dataset_info in dataset_dict.values():
        stage_records = []
        dataset_info["stage_ratio"] = np.array(dataset_info["stage_ratio"])
        dataset_info["stage_ratio"] = dataset_info["stage_ratio"] / \
            sum(dataset_info["stage_ratio"])

        for ratio in dataset_info["stage_ratio"]:
            records = int(
                ratio/sum(dataset_info["stage_ratio"])*dataset_info["n_records"])
            stage_records.append(records)

        dataset_info["records_per_stage"] = stage_records


def get_total_records(dataset_dict):
    total_records = 0
    for dataset_info in dataset_dict.values():
        # print(dataset_dict)
        total_records += dataset_info["n_records"]

    return total_records

def dict_to_df(data):
    # DataFrame用のリストを作成
    # データを構造化するためのリストを準備
    structured_data = []

    # データ構造を変換
    for source, details in data.items():
        for stage, records in enumerate(details['records_per_stage'], start=1):
            structured_data.append({'Source': source, 'Stage': f'Stage {stage}', 'Records': records})

    # リストをDataFrameに変換
    df = pd.DataFrame(structured_data)

    # ピボットテーブルを作成：ステージが行、ソースが列、レコード数が値
    pivot_df = df.pivot(index='Stage', columns='Source', values='Records')
    pivot_df['Sum'] = pivot_df.sum(axis=1)


    # 各ステージの割合を計算する新しいDataFrameを作成
    percentage_df = pivot_df.div(pivot_df['Sum'], axis=0) * 100

    # 'Sum'列を削除して、割合のみのDataFrameを作成
    percentage_df = percentage_df.drop(columns=['Sum'])

    return pivot_df,percentage_df