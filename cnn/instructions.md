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
  
English:
```
$ python3 cnn_text_classifier.py --phase train --language en --path^C./model_state_dict.pth"
  
There are 1 GPU(s) available.
Device name: NVIDIA GeForce RTX 3070 Laptop GPU
	Total loss 4522.697041166015
	Accuracy: 0.5836
	F1 (macro): 0.5854
	F1 (per class): [0.5537618  0.56438356 0.63816003]
               Entailment  Neutral  Contradiction
Entailment            909      532            229
Neutral               437     1030            203
Contradiction         267      418            985

```
  
Spanish:
```
$ python3 cnn_text_classifier.py --phase test --language es --path ./model_state_dict.pth
  
There are 1 GPU(s) available.
Device name: NVIDIA GeForce RTX 3070 Laptop GPU
	Total loss 5875.207520765427
	Accuracy: 0.4752
	F1 (macro): 0.4450
	F1 (per class): [0.26435247 0.54240363 0.52809991]
               Entailment  Neutral  Contradiction
Entailment            297      904            469
Neutral               138     1196            336
Contradiction         142      640            888
```
  
Chinese:
```
$ python3 cnn_text_classifier.py --phase test --language zh --path ./model_state_dict.pth 
  
There are 1 GPU(s) available.
Device name: NVIDIA GeForce RTX 3070 Laptop GPU
Building prefix dict from the default dictionary ...
Dumping model to file cache /tmp/jieba.cache
Loading model cost 0.741 seconds.
Prefix dict has been built successfully.
	Total loss 5482.080330729485
	Accuracy: 0.3623
	F1 (macro): 0.3070
	F1 (per class): [0.43670462 0.07254464 0.41184316]
               Entailment  Neutral  Contradiction
Entailment            978       28            664
Neutral               962       65            643
Contradiction         869       29            772
```