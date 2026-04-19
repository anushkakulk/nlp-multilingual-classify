## CNN for Sentence Classification

Instructions:
1. download the embeddings using the script in `data/download_embeddings.py`, and move them to this directory.
2. download the english training set from [here](https://www.kaggle.com/datasets/thedevastator/xnli-multilingual-nli-dataset?resource=download&select=en_train.csv). Now there should be a file called `en_train.csv'` in this directory as well.
3. train the model
4. evaluate the model
**Note**: report performance achieved with 'model_state_dict.pth', which is included here. It also required the use of full FastText aligned embeddings, not just the top 200,000. However, results were quite similar when using the pre-trained model state dict on the top 200,000 FastText embeddings when compared to the full embeddings.

**Note**: With the top 200,000 english embeddings and a laptop with a GeForce 3070, the training phase took ~20 minutes.

  
For example, from this directory:
```
# 1. get top200000 embeddings. Note that the performance in the paper was achieved with all embeddings, not just the top 200,000 per language.
  
$ python3 ../data/download_embeddings.py --out_dir ./

# 2. download the english training set
# Navigate to the site, and download en_train.csv
# Then, move the resulting .zip here and unzip
$ mv ~/Downloads/en_train.csv.zip
$ unzip en_train.csv.zip

# 3. run the script to train the model
$ python3 cnn_text_classifier.py --phase train --embed_suffix top200000.vec --model_path model_trained.pth

# 4. Evaluate
# Uses included pre-trained dict
$ python3 cnn_text_classifier.py --phase test --language es --embed_suffix top200000.vec

```

#### Training
Example:
```
python3 cnn_text_classifier.py --phase train --language en --model_path "./model_state_dict.pth"
```
This will save the model state to a file called `model_state_dict.pth` after training.

#### Evaluation

Usage example: this loads in the model from a file called `model_state_dict.pth` before evaluating.
  
English:
```
$ python3 cnn_text_classifier.py --phase train --language en --model_path^C./model_state_dict.pth"
  
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
$ python3 cnn_text_classifier.py --phase test --language es --model_path ./model_state_dict.pth
  
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
$ python3 cnn_text_classifier.py --phase test --language zh --model_path ./model_state_dict.pth 
  
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