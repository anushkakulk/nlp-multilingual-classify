import argparse
import jieba
import pandas as pd
import re
from tqdm import tqdm


tqdm.pandas()


def tokenize_zh(text):
    """Strip any existing spacing, then retokenize with MicroTokenizer for uniform Chinese text."""
    if not isinstance(text, str):
        return text

    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)

    text = text.replace(" ", "")
    tokens = jieba.cut(text)
    return " ".join(tokens)


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize Chinese NLI data using MicroTokenizer."
    )
    parser.add_argument("input", help="Path to input CSV (premise, hypothesis, label)")
    parser.add_argument("output", help="Path to save tokenized CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    total = len(df)
    print(f"Loaded {total} rows. Tokenizing...\n")

    print("Tokenizing premises...")
    df["premise"] = df["premise"].progress_apply(tokenize_zh)

    print("Tokenizing hypotheses...")
    df["hypothesis"] = df["hypothesis"].progress_apply(tokenize_zh)

    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\nSaved {total} rows to {args.output}")
    print(df.head(3).to_string())


if __name__ == "__main__":
    main()
