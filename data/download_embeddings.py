import argparse
import os
import urllib.request

EMBEDDING_URLS = {
    "en": "https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.en.align.vec",
    "es": "https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.es.align.vec",
    "zh": "https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.zh.align.vec",
}

def download(lang, out_dir, top_n):
    url = EMBEDDING_URLS[lang]
    filename = f"wiki.{lang}.align.vec" if not top_n else f"wiki.{lang}.align.top{top_n}.vec"
    out_path = os.path.join(out_dir, filename)

    if os.path.exists(out_path):
        print(f"[{lang}] Already exists at {out_path}, skipping.")
        return

    print(f"[{lang}] Downloading from {url} ...")

    if top_n:
        # Stream and keep only the top N lines (header + top_n word vectors)
        with urllib.request.urlopen(url) as response:
            with open(out_path, "w", encoding="utf-8") as f:
                for i, line in enumerate(response):
                    if i == 0:
                        # Rewrite header with new count
                        f.write(f"{top_n} 300\n")
                    elif i <= top_n:
                        f.write(line.decode("utf-8"))
                    else:
                        break
        print(f"[{lang}] Saved top {top_n} vectors to {out_path}")
    else:
        # Download full file with progress
        def progress(block, block_size, total):
            downloaded = block * block_size
            pct = downloaded / total * 100 if total > 0 else 0
            print(f"\r[{lang}] {pct:.1f}% ({downloaded // 1_000_000} MB)", end="")
        urllib.request.urlretrieve(url, out_path, reporthook=progress)
        print(f"\n[{lang}] Saved to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Download aligned fastText embeddings.")
    parser.add_argument("--langs", nargs="+", choices=["en", "es", "zh"], default=["en", "es", "zh"],
                        help="Languages to download (default: all three)")
    parser.add_argument("--out_dir", default="embeddings",
                        help="Directory to save embeddings (default: embeddings/)")
    parser.add_argument("--top_n", type=int, default=200000,
                        help="Only keep top N most frequent vectors to save space (default: 200000, set 0 for full)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    top_n = args.top_n if args.top_n > 0 else None

    for lang in args.langs:
        download(lang, args.out_dir, top_n)

    print("\nDone.")

if __name__ == "__main__":
    main()