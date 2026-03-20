import argparse
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

MODEL_NAME = "xlm-roberta-base"
NUM_LABELS = 3
MAX_LEN = 128


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
    return acc, f1


def train(model, train_loader, val_loader, device, epochs, lr):
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += outputs.loss.item()

        avg_loss = total_loss / len(train_loader)
        acc, f1 = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch + 1} | Loss: {avg_loss:.4f} | Val Acc: {acc:.4f} | Val F1: {f1:.4f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune XLM-RoBERTa on English NLI data."
    )
    parser.add_argument("--train", required=True, help="Path to English train CSV")
    parser.add_argument("--val", required=True, help="Path to English validation CSV")
    parser.add_argument(
        "--resume", default=None, help="Path to saved model weights to resume from"
    )
    parser.add_argument(
        "--output",
        default="xlm_roberta_finetuned.pt",
        help="Path to save model weights",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed from {args.resume}")
    model.to(device)

    train_df = pd.read_csv(args.train, on_bad_lines="skip", engine="python")
    val_df = pd.read_csv(args.val, on_bad_lines="skip", engine="python")

    train_loader = DataLoader(
        NLIDataset(train_df, tokenizer), batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(NLIDataset(val_df, tokenizer), batch_size=args.batch_size)

    train(model, train_loader, val_loader, device, args.epochs, args.lr)

    torch.save(model.state_dict(), args.output)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
