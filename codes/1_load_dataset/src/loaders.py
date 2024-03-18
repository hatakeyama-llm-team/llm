
from datasets import load_dataset
from .CleanedJapaneseWikiDataset import CleanedJapaneseWikiDataset

streaming = True


def wiki_ja_loader():
    return load_dataset("hpprc/wikipedia-20240101", split="train",
                        streaming=streaming
                        ).shuffle()


def wiki_en_loader():
    # 英語
    return load_dataset("wikipedia", "20220301.en", split="train",
                        streaming=streaming,
                        ).shuffle()


def mc4_ja_part_loader():
    return load_dataset("izumi-lab/mc4-ja", split='train',
                        # data_files="data/train-00000-of-00416-a74a40664a952804.parquet",
                        streaming=streaming,
                        )

def cleaned_wiki_loader():
    #クリーンされたjapanese wikipedia
    return CleanedJapaneseWikiDataset(streaming=streaming)

