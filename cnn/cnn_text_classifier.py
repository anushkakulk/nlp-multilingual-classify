import fasttext
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


TRAIN_SET='en_train.csv'
TEST_SET='en_test.csv'
VAL_SET='en_validation.csv'

ft = fasttext.load_model('cc.en.300.bin')
train = pd.read_csv(TRAIN_SET)
val = pd.read_csv(VAL_SET)
test = pd.read_csv(TEST_SET)

"""
word_vector = model.get_word_vector('apple')
sentence_vector = model.get_sentence_vector('Paris is the capital of France')
similar_words = model.get_nearest_neighbors('apple')
# This will return a list of (similarity_score, word) tuples


"""

class CNN(nn.Module):
    def __init__(self, dim: int = 300, num_classes: int = 3, num_filters: int = 100):
        super(CNN, self).__init__()
        self.dim = dim # dimension of word embeddings in sentence
        self.num_filters = num_filters
        self.min_len = 5

        # self.conv_layer = nn.Conv2d(in_channels=1, out_channels=num_filters,  kernel_size=(5, dim), padding=2)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, kernel_size=(h, dim)) for h in (3, 4, 5)
        ])
        self.drop = nn.Dropout(0.5)
        # num filters * len of filters list
        self.connected_layer = nn.Linear(in_features=num_filters * 3, out_features=num_classes) # TODO dimensions

    def embed_sentence(self, sentence: str):
        words = sentence.split()
        embeddings = [ft.get_word_vector(w) for w in words]

        while len(embeddings) < self.min_len:
            embeddings.append(np.zeros(self.dim, dtype=np.float32))
        return torch.tensor(np.array(embeddings))

    def forward(self, sentence_batch: list[str]):
        embeddings = [self.embed_sentence(sentence) for sentence in sentence_batch]
        padded_embeddings = torch.zeros(len(embeddings), max(e.size(0) for e in embeddings), self.dim)
        for idx, embed in enumerate(embeddings):
            padded_embeddings[idx, :embed.size(0), :] = embed


        e = padded_embeddings.unsqueeze(1)
        # output is 0, 1, or 2
        conv_results = []
        for conv in self.convs:
            x = conv(e)
            x = F.relu(x)
            x = x.squeeze(3)
            y = F.max_pool1d(x, x.size(2)).squeeze(2)
            conv_results.append(y)

        z = torch.cat(conv_results, dim=1)
        # x = self.connected_layer(self.drop(self.adaptive_max_pool(x)))
        return self.connected_layer(self.drop(z))

    def evaluate(self, sentence: str):
        self.eval()
        with torch.no_grad():
            return self.forward([sentence])


def main():
    model = CNN()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)

    print("Training...")
    num_epochs = 10
    total = len(train)
    batch = []
    batch_labels = []
    batch_len = 32
    for epoch in range(num_epochs):
        idx = 0
        for row in train.itertuples():
            # access like row.label, row.premise, row.hypothesis
            label = row.label
            # TODO tokenization method?
            sentence = row.premise + " " + row.hypothesis

            batch.append(sentence)
            batch_labels.append(label)

            if len(batch) == batch_len:

                optimizer.zero_grad()
                pred = model.forward(batch)
                loss = loss_fn(pred, torch.tensor(batch_labels, dtype=torch.long))
                loss.backward()
                optimizer.step()

                batch = []
                batch_labels = []
                print(f"[{epoch}] {idx}/{total}")
                
            idx += 1


    print("Testing...")
    total_loss = 0.0
    num_correct = 0
    total = 0
    for row in test.itertuples():
        label = row.label
        # TODO tokenization method?
        sentence = row.premise + row.hypothesis

        pred = model.evaluate(sentence)
        loss = loss_fn(pred, torch.tensor([label]))
        total_loss += loss.item()
        pred = pred.argmax(dim=1).item()
        num_correct += int(pred == label)
        total += 1
    
    print(f"TOTAL LOSS {total_loss}")
    print(f"ACCURACY {num_correct / total}")

if __name__ == "__main__":
    main()