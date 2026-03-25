## CNN for Sentence Classification

Instructions:
1. download the embeddings using the script in `data/download_embeddings.py`, and move them to this directory.
2. download the english training set from [here](https://www.kaggle.com/datasets/thedevastator/xnli-multilingual-nli-dataset?resource=download&select=en_train.csv). Now there should be a file called `en_train.csv'` in this directory as well.
3. train the model
4. evaluate the model

#### Training
Example:
```
python3 cnn_text_classifier.py --phase train --language en --path "./model_state_dict.pth"
```
This will save the model state to a file called `model_state_dict.pth` after training.

#### Evaluation

Usage example: this loads in the model from a file called `model_state_dict.pth` before evaluating.
```
(00:34:34) <0> [~/cs4120/final/nlp-multilingual-classify/cnn]
dani@capyhacker (dm/cnn-initial-impl*+) 🧀❤️ 🐀 $ python3 cnn_text_classifier.py --phase test --language en --path model_state_dict.pt
/home/dani/.local/lib/python3.10/site-packages/jieba/_compat.py:18: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  import pkg_resources
There are 1 GPU(s) available.
Device name: NVIDIA GeForce RTX 3070 Laptop GPU
	Total loss 4522.697041166015
	Accuracy: 0.5836327345309381
	F1 (macro): 0.5854
	F1 (per class): [0.5537618  0.56438356 0.63816003]
dani@capyhacker (dm/cnn-initial-impl*+) 🧀❤️ 🐀 $ python3 cnn_text_classifier.py --phase test --language es --path model_state_dict.pth
/home/dani/.local/lib/python3.10/site-packages/jieba/_compat.py:18: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  import pkg_resources
There are 1 GPU(s) available.
Device name: NVIDIA GeForce RTX 3070 Laptop GPU
	Total loss 5875.207520765427
	Accuracy: 0.475249500998004
	F1 (macro): 0.4450
	F1 (per class): [0.26435247 0.54240363 0.52809991]
(00:39:13) <0> [~/cs4120/final/nlp-multilingual-classify/cnn]
dani@capyhacker (dm/cnn-initial-impl*+) 🧀❤️ 🐀 $ python3 cnn_text_classifier.py --phase test --language zh --path model_state_dict.pth
/home/dani/.local/lib/python3.10/site-packages/jieba/_compat.py:18: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  import pkg_resources
There are 1 GPU(s) available.
Device name: NVIDIA GeForce RTX 3070 Laptop GPU
Building prefix dict from the default dictionary ...
Dumping model to file cache /tmp/jieba.cache
Loading model cost 0.830 seconds.
Prefix dict has been built successfully.
	Total loss 5482.080330729485
	Accuracy: 0.36227544910179643
	F1 (macro): 0.3070
	F1 (per class): [0.43670462 0.07254464 0.41184316]
(00:41:14) <0> [~/cs4120/final/nlp-multilingual-classify/cnn]

```