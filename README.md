# NLP Project: Examining Cross-Lingual Transfer for Document Classification

This project investigates how well NLP models trained on English can transfer to other languages without multilingual training data, using the task of Natural Language Inference (NLI).

## Data

Dataset: [XNLI (Cross-lingual Natural Language Inference)](https://huggingface.co/datasets/facebook/xnli)

**Languages:**
- English
- Spanish
- Chinese — tokenized with [jieba](https://github.com/fxsjy/jieba)

English is used for training. Spanish and Chinese are used for cross-lingual evaluation only. See `data/README.md` for details on data format, splits, and preprocessing.

## Models

**TF-IDF + Logistic Regression**
Baseline model using TF-IDF vectors built from concatenated premise-hypothesis pairs, classified with multinomial logistic regression.

**CNN with fastText Embeddings**
Convolutional neural network using pretrained aligned fastText word embeddings. Aligned vectors allow the model trained on English to directly process Spanish and Chinese at test time.

**XLM-RoBERTa**
Pretrained multilingual transformer fine-tuned on English NLI data. Cross-lingual performance on Spanish and Chinese is evaluated without any target-language fine-tuning.

## Evaluation

All models are evaluated on Spanish and Chinese test sets using:
- Accuracy
- Macro F1 score
- Confusion matrix