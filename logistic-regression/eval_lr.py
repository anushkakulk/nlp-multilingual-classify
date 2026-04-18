import argparse
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def combine_sentences(df):
    premise = df["premise"].fillna("")
    hypothesis = df["hypothesis"].fillna("")
    return (premise + " " + hypothesis).values

def evaluate(language, df, model): 
    X_combined = combine_sentences(df)
    y = df["label"].values
    y_pred = model.predict(X_combined)

    print(f"\n---- {language} METRICS ----")
    print("Accuracy:", accuracy_score(y, y_pred))
    print("F1-Score:", f1_score(y, y_pred, average="macro"))

    confusion = confusion_matrix(y, y_pred)
    print("Confusion Matrix: \n", confusion)

    disp = ConfusionMatrixDisplay(confusion_matrix = confusion)
    disp.plot(cmap="Blues", values_format="d")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Evaluate saved TF-IDF and LR model")
    parser.add_argument("--model", default="models/lr_model.joblib")
    parser.add_argument("--en_test", default=None, help="Path to English test CSV")
    parser.add_argument("--es_test", default=None, help="Path to Spanish test CSV")
    parser.add_argument("--zh_test", default=None, help="Path to Chinese test CSV")
    args = parser.parse_args()

    lr_model = joblib.load(args.model)

    if args.en_test:
        en_df = pd.read_csv(args.en_test, on_bad_lines="skip", engine="python")
        evaluate("English Test", en_df, lr_model)

    if args.es_test:
        es_df = pd.read_csv(args.es_test, on_bad_lines="skip", engine="python")
        evaluate("Spanish Test", es_df, lr_model)

    if args.zh_test:
        zh_df = pd.read_csv(args.zh_test, on_bad_lines="skip", engine="python")
        evaluate("Chinese Test", zh_df, lr_model)

if __name__ == "__main__":
    main()

