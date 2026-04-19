import fasttext
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import jieba
import re
import os
import nltk
from nltk.tokenize import word_tokenize
import argparse
from gensim.models import KeyedVectors
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


ENGLISH_VAL_SET='../data/english/en_validation.csv'
LABEL_NAMES = ["Entailment", "Neutral", "Contradiction"]
LANGUAGE_NAMES = {
    "en": "english",
    "es": "spanish",
    "zh": "chinese"
}

"""
Reference implementation:
https://chriskhanhtran.github.io/posts/cnn-sentence-classification/

"""

if torch.cuda.is_available():       
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


def tokenize_zh(text):
    """Strip any existing spacing, then retokenize with MicroTokenizer for uniform Chinese text."""
    if not isinstance(text, str):
        return text

    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)

    text = text.replace(" ", "")
    tokens = jieba.cut(text)
    return " ".join(tokens)


def tokenize(sentence: str, language: str):
    if language == "en":
        return word_tokenize(sentence)
    elif language == "es":
        return word_tokenize(sentence)
    elif language == "zh":
        return tokenize_zh(sentence)
    else:
        return []
    

class CNN(nn.Module):
    def __init__(self, embeddings, dim: int = 300, num_classes: int = 3, num_filters: int = 100):
        super(CNN, self).__init__()
        self.word_embeddings = embeddings
        self.filter_sizes = [3, 4, 5]
        self.dim = dim
        self.num_filters = num_filters
        self.min_len = 5
        self.encoded_dim = num_filters * len(self.filter_sizes)
        self.train_language = "en"

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=self.dim, out_channels=num_filters, kernel_size=self.filter_sizes[i]) for i in range(len(self.filter_sizes))
        ])
        self.drop = nn.Dropout(0.5)
        self.connected_layer = nn.Linear(in_features=self.encoded_dim * 2, out_features=num_classes)

    def embed_sentence(self, sentence: str, language: str):
        words = tokenize(sentence, language)
        embeddings = []
        for w in words:
            if w in self.word_embeddings:
                embeddings.append(self.word_embeddings[w])
            else:
                embeddings.append(np.zeros(self.dim, dtype=np.float32))

        while len(embeddings) < self.min_len:
            embeddings.append(np.zeros(self.dim, dtype=np.float32))
        return torch.tensor(np.array(embeddings))

    def forward(self, premise_batch: list[str], hypothesis_batch: list[str], language: str = "en"):
        premise_embeds = [self.embed_sentence(premise, language=language) for premise in premise_batch]
        hypothesis_embeds = [self.embed_sentence(hypothesis, language=language) for hypothesis in hypothesis_batch]

        padded_premise = torch.zeros(len(premise_embeds), max(e.size(0) for e in premise_embeds), self.dim).to(device)
        for idx, embed in enumerate(premise_embeds):
            padded_premise[idx, :embed.size(0)] = embed.to(device)
        padded_hypothesis = torch.zeros(len(hypothesis_embeds), max(e.size(0) for e in hypothesis_embeds), self.dim).to(device)
        for idx, embed in enumerate(hypothesis_embeds):
            padded_hypothesis[idx, :embed.size(0)] = embed.to(device)

        premise_reshaped = padded_premise.permute(0, 2, 1)
        hypothesis_reshaped = padded_hypothesis.permute(0, 2, 1)

        premise_results = []
        hypothesis_results = []
        for conv in self.convs:
            x = conv(premise_reshaped)
            x = F.relu(x)
            x = F.max_pool1d(x, x.size(2)).squeeze(2)
            premise_results.append(x)

        for conv in self.convs:
            x = conv(hypothesis_reshaped)
            x = F.relu(x)
            x = F.max_pool1d(x, x.size(2)).squeeze(2)
            hypothesis_results.append(x)


        z1 = torch.cat(premise_results, dim=1)
        z2 = torch.cat(hypothesis_results, dim=1)

        z = torch.cat([z1, z2], dim=1)
        return self.connected_layer(self.drop(z))

    def evaluate(self, premise_batch: list[str], hypothesis_batch: list[str], language: str = "en"):
        self.eval()
        with torch.no_grad():
            return self.forward(premise_batch, hypothesis_batch, language=language)


def train_model(model, loss_fn, optimizer, train_path):
    model.to(device)
    print("Training...")
    num_epochs = 8
    premise_batch = []
    hypothesis_batch = []
    batch_labels = []
    batch_len = 32
    batch_count = 0

    train = pd.read_csv(train_path)
    val = pd.read_csv(ENGLISH_VAL_SET)

    for epoch in range(num_epochs):
        idx = 0
        train_shuffled = train.sample(frac=1).reset_index(drop=True)
        model.train()

        for row in train_shuffled.itertuples():
            label = row.label
            premise_batch.append(row.premise)
            hypothesis_batch.append(row.hypothesis)
            batch_labels.append(label)
            batch_count += 1

            if batch_count == batch_len:

                optimizer.zero_grad()
                pred = model.forward(premise_batch, hypothesis_batch)
                loss = loss_fn(pred, torch.tensor(batch_labels, dtype=torch.long).to(device))
                loss.backward()
                optimizer.step()

                batch_labels = []
                hypothesis_batch = []
                premise_batch = []
                batch_count = 0
                
            idx += 1

        model.eval()
        val_total = 0
        val_total_loss = 0.0
        val_correct = 0
        print(f"\tValidation for epoch {epoch}")
        for row in val.itertuples():
            label = row.label
            pred = model.evaluate([row.premise], [row.hypothesis])
            loss = loss_fn(pred, torch.tensor([label]).to(device))
            val_total_loss += loss.item()
            pred = pred.argmax(dim=1).item()
            val_correct += int(pred == label)
            val_total += 1
        print(f"\tTotal loss {val_total_loss} accuracy {val_correct / val_total}")

def test_model(model, loss_fn, language):
    total_loss = 0.0
    num_correct = 0
    total = 0
    all_preds = []
    all_labels = []

    model.to(device)
    model.eval()
    test = pd.read_csv(f'../data/{LANGUAGE_NAMES[language]}/{language}_test.csv')
    for row in test.itertuples():
        label = row.label
        pred = model.evaluate([row.premise], [row.hypothesis], language)
        loss = loss_fn(pred, torch.tensor([label]).to(device))
        total_loss += loss.item()
        pred = pred.argmax(dim=1).item()
        num_correct += int(pred == label)
        all_preds.append(pred)
        all_labels.append(label)
        total += 1
    
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_per_class = f1_score(all_labels, all_preds, average=None)

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\tTotal loss {total_loss}")
    print(f"\tAccuracy: {acc:.4f}")
    print(f"\tF1 (macro): {f1_macro:.4f}")
    print(f"\tF1 (per class): {f1_per_class}")
    print(pd.DataFrame(cm, index=LABEL_NAMES, columns=LABEL_NAMES))


def main():
    parser = argparse.ArgumentParser(description="A script to train or evaluate a CNN for text classification")
    parser.add_argument('--phase', required=True, help="Phase of model, either 'train' or 'test'")
    parser.add_argument('--train_path', required=False, default='../data/en_train.csv', help="Path to english training data.")
    parser.add_argument("--embed_suffix", required=False, default='', help="Suffix of aligned embedding files. For example, for 'wiki.en.align.top200000.vec', pass in 'top200000.vec'. For wiki.en.align.vec, do not pass in this argument.")
    parser.add_argument('--language', required=False, default="en", help="Language to evaluate model on, used during 'test'. Either 'en' (English), 'es' (Spanish), or 'zh' (Chinese).")
    parser.add_argument('--model_path', required=False, default='model_trained.pth', help="Path to pre-trained model. Defaults to 'model_state_dict.pth'. Used during 'test'.")
    args = parser.parse_args()
    phase = args.phase
    train_path = args.train_path
    embed_suffix = args.embed_suffix
    language = args.language
    pretrained_path = args.model_path


    if phase == "train":
        if not os.path.exists(f'wiki.en.align.{embed_suffix}'):
            print("Must download English FastText aligned embeddings! See instructions in 'instructions.md'")
            exit(-1)

        if not os.path.exists(train_path):
            print("Must download English training set! See instructions in 'instructions.md'")
            exit(-1)

        if not os.path.exists(train_path):
            print("Must pass in a correct path for the english training data!")
            exit(-1)

        ft_en = KeyedVectors.load_word2vec_format(f'wiki.en.align.{embed_suffix}', binary=False, limit=200000)
        model = CNN(ft_en)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adadelta(model.parameters(), lr=0.1, rho=0.95)
        train_model(model, loss_fn, optimizer, train_path)
        torch.save(model.state_dict(), pretrained_path)
    elif phase == "test":
        pretrained_state_dict = torch.load(pretrained_path, weights_only=True)
        if language == "en":
            if not os.path.exists(f'wiki.en.align.{embed_suffix}'):
                print("Must download English FastText aligned embeddings! See instructions in 'instructions.md'")
                exit(-1)
            ft_en = KeyedVectors.load_word2vec_format(f'wiki.en.align.{embed_suffix}', binary=False, limit=200000)
            model = CNN(ft_en)
            model.load_state_dict(pretrained_state_dict)
            loss_fn = nn.CrossEntropyLoss()
            test_model(model, loss_fn, "en")
        elif language == "es":
            if not os.path.exists(f'wiki.es.align.{embed_suffix}'):
                print("Must download Spanish FastText aligned embeddings! See instructions in 'instructions.md'")
                exit(-1)
            ft_es = KeyedVectors.load_word2vec_format(f'wiki.es.align.{embed_suffix}', binary=False, limit=200000)
            model = CNN(ft_es)
            model.load_state_dict(pretrained_state_dict)
            loss_fn = nn.CrossEntropyLoss()
            test_model(model, loss_fn, "es")
        elif language == "zh":
            if not os.path.exists(f'wiki.zh.align.{embed_suffix}'):
                print("Must download Chinese FastText aligned embeddings! See instructions in 'instructions.md'")
                exit(-1)
            ft_zh = KeyedVectors.load_word2vec_format(f'wiki.zh.align.{embed_suffix}', binary=False, limit=200000)
            model = CNN(ft_zh)
            model.load_state_dict(pretrained_state_dict)
            loss_fn = nn.CrossEntropyLoss()
            test_model(model, loss_fn, "zh")
        

if __name__ == "__main__":
    main()