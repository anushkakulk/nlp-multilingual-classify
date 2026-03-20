import argparse
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

MODEL_NAME = "xlm-roberta-base"
NUM_LABELS = 3
MAX_LEN = 128
LABEL_NAMES = ["Entailment", "Neutral", "Contradiction"]


class NLIDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.encodings = tokenizer(
            list(df["premise"]),
            list(df["hypothesis"]),
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.labels = torch.tensor(df["label"].values, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    cm = confusion_matrix(all_labels, all_preds)
    return acc, f1, cm


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned XLM-RoBERTa on target languages."
    )
    parser.add_argument(
        "--model", required=True, help="Path to saved model weights (.pt)"
    )
    parser.add_argument("--test_en", required=True, help="Path to English test CSV")
    parser.add_argument("--test_es", required=True, help="Path to Spanish test CSV")
    parser.add_argument("--test_zh", required=True, help="Path to Chinese test CSV")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)

    for name, path in [
        ("English", args.test_en),
        ("Spanish", args.test_es),
        ("Chinese", args.test_zh),
    ]:
        df = pd.read_csv(path)
        loader = DataLoader(NLIDataset(df, tokenizer), batch_size=args.batch_size)
        acc, f1, cm = evaluate(model, loader, device)
        print(f"\n--- {name} Test Results ---")
        print(f"Accuracy: {acc:.4f}")
        print(f"Macro F1: {f1:.4f}")
        print("Confusion Matrix:")
        print(pd.DataFrame(cm, index=LABEL_NAMES, columns=LABEL_NAMES))


if __name__ == "__main__":
    main()
