import argparse
import joblib
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold

def combine_sentences(df):
    premise = df["premise"].fillna("")
    hypothesis = df["hypothesis"].fillna("")
    return (premise + " " + hypothesis).values


def train(train_df):
    X_train = combine_sentences(train_df)
    y_train = train_df["label"].values

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
        min_df = 2, 
        ngram_range = (1,3),
        sublinear_tf=True
        )), 
        ("lr", LogisticRegression(
        max_iter=2000,
        solver="saga",
        C = 1.0
        ))
    ])

    param_grid = {
        "tfidf__max_features": [50_000, 100_000, 200_000, 300_000]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=pipe, 
        param_grid=param_grid, 
        scoring="f1_macro",
        cv=cv,
        verbose=1,
    )

    grid.fit(X_train,y_train)

    print("Best params:", grid.best_params_)
    print("Best CV f1_macro", grid.best_score_)

    best_model = grid.best_estimator_

    return best_model

def evaluate_en(df, model):
    X_val = combine_sentences(df)
    y_val = df["label"].values
    pred_values = model.predict(X_val)

    print("English Accuracy:", accuracy_score(y_val, pred_values))
    print("English F1-Score:", f1_score(y_val, pred_values, average="macro"))

def main():
    parser = argparse.ArgumentParser(description="Train TF-IDF + LR on English NLI Data")
    parser.add_argument("--train", required=True, help="Path to english train csv")
    parser.add_argument("--val", default=None, help="Validation for english training")
    parser.add_argument("--vectorizer_output", default="models/vectorizer.joblib")
    parser.add_argument("--model_output", default="models/lr_model.joblib")
    args = parser.parse_args()

    train_df = pd.read_csv(args.train, on_bad_lines="skip", engine="python")
    model = train(train_df)
    print("\n----TRAIN METRICS----")
    evaluate_en(train_df, model)

    if (args.val):
        val_df = pd.read_csv(args.val, on_bad_lines="skip", engine="python")
        print("\n----VALIDATION METRICS----")
        evaluate_en(val_df, model)
    
    joblib.dump(model, args.model_output)

    print(f"Model saved to: {args.model_output}")

if __name__ == "__main__":
    main()








    
