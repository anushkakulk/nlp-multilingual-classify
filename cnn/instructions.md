## CNN for Sentence Classification

Instructions:
1. download the embeddings using the script in `data/download_embeddings.py`, and move them to this directory.
2. download the english training set from [here](https://www.kaggle.com/datasets/thedevastator/xnli-multilingual-nli-dataset?resource=download&select=en_train.csv). Now there should be a file called `en_train.csv'` in this directory as well.
3. train the model
4. evaluate the model

#### Training
Example:
```
python3 cnn_text_classifier.py --phase train --language en --path "./model_state-dict"
```
This will save the model state to a file called `model_state-dict` after training.

#### Evaluation

Usage example: this loads in the model from a file called `model_state-dict` before evaluating.
```
(20:21:33) <0> [~/cs4120/final/nlp-multilingual-classify/cnn]
dani@capyhacker (dm/cnn-initial-impl*+) 🧀❤️ 🐀 $ python3 cnn_text_classifier.py --phase test --language en --path "./model_state-dict"
There are 1 GPU(s) available.
Device name: NVIDIA GeForce RTX 3070 Laptop GPU
	Total loss 4593.350525464863
	Accuracy: 0.5728542914171657
	F1 (macro): 0.5739
	F1 (per class): [0.54496403 0.54140127 0.63529412]
(20:23:43) <0> [~/cs4120/final/nlp-multilingual-classify/cnn]
dani@capyhacker (dm/cnn-initial-impl*+) 🧀❤️ 🐀 $ python3 cnn_text_classifier.py --phase test --language es --path "./model_state-dict"
There are 1 GPU(s) available.
Device name: NVIDIA GeForce RTX 3070 Laptop GPU
	Total loss 5861.355050156417
	Accuracy: 0.43832335329341315
	F1 (macro): 0.4085
	F1 (per class): [0.26672421 0.42861831 0.53006873]
(20:26:07) <0> [~/cs4120/final/nlp-multilingual-classify/cnn]
dani@capyhacker (dm/cnn-initial-impl*+) 🧀❤️ 🐀 $ python3 cnn_text_classifier.py --phase test --language zh --path "./model_state-dict"
There are 1 GPU(s) available.
Device name: NVIDIA GeForce RTX 3070 Laptop GPU
Building prefix dict from the default dictionary ...
Loading model from cache /tmp/jieba.cache
Loading model cost 1.825 seconds.
Prefix dict has been built successfully.
	Total loss 5485.861373245716
	Accuracy: 0.3652694610778443
	F1 (macro): 0.2891
	F1 (per class): [0.29556289 0.07799443 0.49363405]
(20:29:18) <0> [~/cs4120/final/nlp-multilingual-classify/cnn]
dani@capyhacker (dm/cnn-initial-impl*+) 🧀❤️ 🐀 $ 
```