
from datasets import load_dataset

noise_strings="""
存命人物
関連人物
脚注
参考文献 
"""

def clean_text(txt):
    txt=txt.replace("\n\n","\n")
    lines=txt.split("\n")
    cleaned_lines=[]
    touten_flag=False

    for line in lines[::-1]:
        if line.endswith("。"):
            touten_flag=True
        if touten_flag:
            cleaned_lines.append(line)

    cleaned_lines=cleaned_lines[::-1]


    noise_list=noise_strings.strip().split("\n")

    cleaned_lines2=[]
    stop_flag=False
    for line in cleaned_lines:
        for word in noise_list:
            if word in line:
                stop_flag=True
        if stop_flag:
            break
        cleaned_lines2.append(line)

    return "\n".join(cleaned_lines2)


class CleanedJapaneseWikiDataset:
    def __init__(self,streaming=True):
        self.dataset=load_dataset("hpprc/wikipedia-20240101", split="train",
                                streaming=streaming
                                ).shuffle()
        self.loader=iter(self.dataset)

    def __iter__(self):
        # イテレータは自分自身を返す
        return self

    def __next__(self):
        d=next(self.loader)
        d["text"]=clean_text(d["text"])
        return d