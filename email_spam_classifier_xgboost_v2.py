"""
Train an email spam classifier using a TF-IDF vectorizer and XGBoost model.

This script:
- Loads the spam/ham email dataset.
- Cleans text data (removes HTML tags and non-alphanumeric characters).
- Builds a scikit-learn Pipeline consisting of:
    - A custom text cleaning step (FunctionTransformer)
    - TF-IDF feature extraction
    - XGBoost classification model
- Configures model and vectorizer hyperparameters.
- Performs 5-fold stratified cross-validation using accuracy, precision, recall, and F1 metrics.
- Logs cross-validation results to a report file in /reports/.
- Fits the final model to the full dataset.
- Optionally prints predictions on random sample emails for inspection.
- Saves the trained model pipeline to /models/ for later use.

Usage:
    python train_email_spam_xgboost.py

Output:
    /reports/email_spam_xgboost.txt      - performance report
    /models/email_classifier_xgboost.joblib   - trained pipeline model
"""

#import re
import re
#import os
import os
#import pandas
import pandas as pd
#import joblib
from joblib import dump
#import pipeline
from sklearn.pipeline import Pipeline
#import function transformer
from sklearn.preprocessing import FunctionTransformer
#import vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
#import cross validation
from sklearn.model_selection import cross_validate, StratifiedKFold
#import xgboost model
from xgboost import XGBClassifier

#import data
DATA_PATH = "data/spam_ham_dataset.csv"
email_df = pd.read_csv(DATA_PATH, encoding="latin-1")
#remove non necessary columns
email_df = email_df[["text", "label_num"]]
#clean email text function
def clean_email_text(emails):
    """
    Clean raw email text by removing HTML tags and non-alphanumeric characters.

    Parameters
    ----------
    emails : list-like or iterable of str
        A collection of raw email text strings to be cleaned.

    Returns
    -------
    list of str
        A list containing the cleaned text for each email. The cleaning process:
        - Strips unnecessary whitespace
        - Removes HTML tags such as <br> or <a href=...>
        - Removes characters that are not letters, numbers, apostrophes, or spaces

    Notes
    -----
    This function is intended for use inside a scikit-learn FunctionTransformer,
    so it accepts and returns iterable text collections rather than single strings.
    """
    clean_emails = []
    for email in emails:
        email = email.strip()
        email = re.sub(r"<.*?>", repl="", string=email) #remove any html tags
        email = re.sub(r"[^A-Za-z0-9'\s]", repl="", string=email) #remove non alfa and num char
        clean_emails.append(email)
    return clean_emails
#create pipeline
email_pipeline = Pipeline([
    ("preprocessor", FunctionTransformer(
        clean_email_text,
        validate=False
    )),
    ("vectorizer", TfidfVectorizer(
    )),
    ("XGBClassifier", XGBClassifier(
    ))
])
#set XGBClassifier parameters
email_pipeline.set_params(**{
    "XGBClassifier__n_estimators": 300,
    "XGBClassifier__learning_rate": 0.1,
    "XGBClassifier__n_jobs": -1,
    "vectorizer__lowercase": True,
    "vectorizer__stop_words": "english",
    "vectorizer__ngram_range": (1,2),
    "vectorizer__max_features": 1000
})
#define cross-validation train/test split
CV = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=0
)
#define cross-validation tests
SCORING = ["accuracy", "precision", "recall", "f1"]
#preform cross-validation
email_cross_validation_score = cross_validate(
    email_pipeline,
    email_df["text"],
    email_df["label_num"],
    cv=CV,
    scoring=SCORING,
    return_train_score=False
)
#paths
REPORT_PATH = "reports/email_spam_xgboost.txt"
MODEL_PATH = "models/email_classifier_xgboost.joblib"
#create folder
os.makedirs("reports", exist_ok=True)
#create report
report_lines=[]
report_lines.append(
    f"fit_time mean±std: {email_cross_validation_score['fit_time'].mean():.3f}"
    f" ± {email_cross_validation_score['fit_time'].std():.3f}s\n"
)
report_lines.append(
    f"score_time mean±std: {email_cross_validation_score['score_time'].mean():.3f}"
    f" ± {email_cross_validation_score['score_time'].std():.3f}s\n\n"
)
for metric in SCORING:
    scores = email_cross_validation_score[f"test_{metric}"]
    mean = scores.mean()
    std = scores.std()
    report_lines.append(
        f"{metric.capitalize():<10} mean±std: {mean:.4f} ± {std:.4f} -> {scores}\n"
    )
#write all metrics to a new report file
with open(REPORT_PATH, mode="w", encoding="utf-8") as report:
    report.writelines(report_lines)
#fit model
email_pipeline.fit(email_df["text"], email_df["label_num"])
#sample predictions
SAMPLE_EMAIL = True
N=5
if SAMPLE_EMAIL:
    sample = email_df.sample(
        N,
        random_state=0
    )
    email_prediction = email_pipeline.predict(sample["text"])
    for email_text, true_label, predicted_label in zip(
        sample["text"],
        sample["label_num"],
        email_prediction
    ):
        TRUE = "SPAM" if true_label == 1 else "HAM"
        PRED = "SPAM" if predicted_label == 1 else "HAM"
        print(f"TRUE: {TRUE:<4} | PRED: {PRED:<4} -> {email_text[:200]}...")
#save fitted pipeline
os.makedirs("models", exist_ok=True)
dump(email_pipeline, MODEL_PATH)
