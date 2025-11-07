# Email Spam Classifier (XGBoost + TF-IDF) — v2

This project trains a machine learning model to classify email messages as **SPAM** or **HAM**.  
Version 2 improves the preprocessing pipeline, vectorization, and XGBoost hyperparameters, resulting in more stable and accurate predictions.

---

## ✨ What’s New in v2

- Improved text cleaning rules
- TF-IDF vectorization with n-grams `(1, 2)`
- XGBoost tuned with `n_estimators=300` & `learning_rate=0.1`
- Reduced vocabulary size using `max_features=1000`
- Cleaner and more portable training script

---

## 🚀 Pipeline Overview
