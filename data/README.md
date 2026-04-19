# Data

This folder contains the [XNLI (Cross-lingual Natural Language Inference) dataset](https://huggingface.co/datasets/facebook/xnli) used for training and evaluating cross-lingual transfer models.

**Note:** In order to download the train data, you will need to use these links:
1. [english train data](https://www.kaggle.com/datasets/thedevastator/xnli-multilingual-nli-dataset?resource=download&select=en_train.csv)
2. [spanish train data](https://www.kaggle.com/datasets/thedevastator/xnli-multilingual-nli-dataset?resource=download&select=es_train.csv)
3. [chinese train data](https://www.kaggle.com/datasets/thedevastator/xnli-multilingual-nli-dataset?resource=download&select=zh_train.csv)

All of these files are too large to commit to the repo. See specific model sub-folders for instructions on where to download these files, and how to feed them to the model scripts.

## Folder Structure

```
tokenize_chinese_text.py
download_embeddings.py
english/
├── en_test.csv
└── en_validation.csv
spanish/
├── es_test.csv
└── es_validation.csv
chinese/
├── zh_test.csv
└── zh_validation.csv
embeddings/
├── wiki.en.align.top200000.vec
├── wiki.es.align.top200000.vec
└── wiki.zh.align.top200000.vec

```
(https://www.kaggle.com/datasets/thedevastator/xnli-multilingual-nli-dataset?resource=download&select=en_train.csv)
## File Format

All CSV files contain the following columns:

| Column | Description |
|---|---|
| `premise` | The base sentence that forms the foundation for the inference task |
| `hypothesis` | The second sentence whose logical relationship to the premise is being classified |
| `label` | Integer label indicating the relationship between premise and hypothesis |

### Labels

| Value | Class | Description |
|---|---|---|
| `0` | Entailment | The hypothesis can be inferred from the premise |
| `1` | Neutral | There is no clear logical relationship between the two sentences |
| `2` | Contradiction | The hypothesis contradicts the premise |

## Dataset Split
| Split | Sentences |
|---|---|
| Train | 392,702 |
| Test | 5,010 |
| Val | 2,490 |
Each split is available in all three  languages with the same sentences translated across languages.


## Languages

- **English (`en`)** — Standard whitespace-tokenized text.
- **Spanish (`es`)** — Standard whitespace-tokenized text.
- **Chinese (`zh`)** — Raw data contains pre-tokenized text with spaces between characters; the tokenization script below should be run before use.

## Word Embeddings
Pretrained aligned fastText embeddings are used for the CNN model. Because the vectors are aligned across languages, they share the same vector space, enabling cross-lingual transfer without any multilingual training data.

The download script retrieves only the top 200,000 most frequent word vectors per language (sorted by frequency in the source corpus)

Run the download script:
```
# Download all 3 languages (recommended)
python download_embeddings.py --out_dir embeddings
```

Embeddings will be saved to embeddings/ as:
```
wiki.en.align.top200000.vec
wiki.es.align.top200000.vec
wiki.zh.align.top200000.vec
```

Note: The embeddings/ folder is not tracked by git due to file size. Run the download script to regenerate it locally.


## Chinese Tokenization

The raw Chinese files may contain inconsistently spaced or unsegmented text. Before using the Chinese data, run the tokenization script to produce clean, consistently segmented output using [jieba](https://github.com/fxsjy/jieba).

**Install jieba:**
```bash
pip install jieba
```

**Run the tokenizer:**
```bash
python tokenize_chinese_text.py chinese/[input_file].csv chinese/[output_file]_tokenized.csv
```

**Example:**
```bash
python tokenize_chinese_text.py chinese/zh_test.csv chinese/zh_test_tokenized.csv
```

The script strips any existing spacing between Chinese characters and re-tokenizes using jieba, producing space-separated tokens suitable for downstream TF-IDF vectorization.