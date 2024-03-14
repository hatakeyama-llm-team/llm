# Appends a path to import python scripts that are in other directories.
from types import SimpleNamespace
import sentencepiece as spm
import sys
import yaml
import os

sys.path.append("../common")


if True:
    from special_token_list import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, CLS_TOKEN, SEP_TOKEN, EOD_TOKEN, MASK_TOKEN, NEWLINE_TOKEN


def yaml_to_namespace(yaml_path):
    with open(yaml_path, 'r') as file:
        # YAMLファイルを辞書として読み込む
        data = yaml.safe_load(file)
        # 辞書をSimpleNamespaceに変換
        return recursive_namespace(data)


def recursive_namespace(data):
    if isinstance(data, dict):
        # 再帰的に辞書の各要素をSimpleNamespaceに変換
        return SimpleNamespace(**{k: recursive_namespace(v) for k, v in data.items()})
    elif isinstance(data, list):
        # リストの要素も変換
        return [recursive_namespace(v) for v in data]
    else:
        # その他のデータ型はそのまま返す
        return data


def main():
    args = yaml_to_namespace('sentence_piece_config.yaml')

    # Trains a SentencePiece tokenizer. After training, *.model and *.vocab will be saved in the current directory.
    spm.SentencePieceTrainer.train(
        input=args.input,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
        model_type=args.model_type,
        num_threads=args.num_threads,
        train_extremely_large_corpus=args.train_extremely_large_corpus,
        user_defined_symbols=[
            BOS_TOKEN,
            EOS_TOKEN,
            PAD_TOKEN,
            CLS_TOKEN,
            SEP_TOKEN,
            EOD_TOKEN,
            MASK_TOKEN,
            NEWLINE_TOKEN,
        ],  # Note: `NEWLINE_TOKEN` is needed in `user_defined_symbols`.
        byte_fallback=True,
        split_digits=True,
        allow_whitespace_only_pieces=True,
        remove_extra_whitespaces=False,
    )

    # move trained file to target dir
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    os.rename(f"{args.model_prefix}.model",
              f"{args.output_dir}/{args.model_prefix}.model")
    os.rename(f"{args.model_prefix}.vocab",
              f"{args.output_dir}/{args.model_prefix}.vocab")


if __name__ == "__main__":
    main()
