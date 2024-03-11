
from datasets import load_dataset


def wiki_ja_loader():
    return load_dataset("hpprc/wikipedia-20240101", split="train").shuffle()


def wiki_en_loader():
    # 英語
    return load_dataset("wikipedia", "20220301.en", split="train").shuffle()
