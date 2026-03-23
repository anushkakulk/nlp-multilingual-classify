import argparse
import joblib
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

def combine_sentences(df):
    premise = df["premise"].fillna("")
    hypothesis = df["hypothesis"].fillna("")
    return (premise + " " + hypothesis).values


def train(train_df):
    X_train = combine_sentences(train_df)
    y_train = train_df["label"].values

    vectorizer = TfidfVectorizer(
        min_df = 2, 
        ngram_range = (1,3),
        max_features=200_000, 
        sublinear_tf=True
    )

    X_train = vectorizer.fit_transform(X_train)

    lr_model = LogisticRegression(
        max_iter=2000,
        solver="saga",
        C = 1.0
    )

    lr_model.fit(X_train, y_train)

    return vectorizer, lr_model

def evaluate_en(df, vectorizer, lr_model):
    X_val = combine_sentences(df)
    y_val = df["label"].values
    X_vect = vectorizer.transform(X_val)
    pred_val = lr_model.predict(X_vect)

    print("English Accuracy:", accuracy_score(y_val, pred_val))
    print("English F1-Score:", f1_score(y_val, pred_val, average="macro"))


def main():
    parser = argparse.ArgumentParser(description="Train TF-IDF + LR on English NLI Data")
    parser.add_argument("--train", required=True, help="Path to english train csv")
    parser.add_argument("--val", default=None, help="Validation for english training")
    parser.add_argument("--vectorizer_output", default="models/vectorizer.joblib")
    parser.add_argument("--model_output", default="models/lr_model.joblib")
    args = parser.parse_args()

    train_df = pd.read_csv(args.train, on_bad_lines="skip", engine="python")
    vectorizer, lr_model = train(train_df)
    print("\n----TRAIN METRICS----")
    evaluate_en(train_df, vectorizer, lr_model)

    if (args.val):
        val_df = pd.read_csv(args.val, on_bad_lines="skip", engine="python")
        print("\n----VALIDATION METRICS----")
        evaluate_en(val_df, vectorizer, lr_model)
    
    joblib.dump(vectorizer, args.vectorizer_output)
    joblib.dump(lr_model, args.model_output)

    print(f"Model saved to: {args.model_output}")
    print(f"Vectorizer saved to: {args.vectorizer_output}")

if __name__ == "__main__":
    main()








    
