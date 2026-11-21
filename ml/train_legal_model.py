
import json
import lzma
import os
from pathlib import Path
from collections import Counter

import joblib
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score
)

# --------------------------------------------------
# 1. DATASET PATHS
# --------------------------------------------------

BASE = Path("court-dataset") / "Layman-new-dataset"

TRAIN_FILE = BASE / "FT-Layman-train.jsonl.xz"
DEV_FILE = BASE / "FT-Layman-dev.jsonl.xz"
TEST_FILE = BASE / "FT-Layman-test.jsonl.xz"

MODEL_DIR = Path("ml") / "artifacts"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "legal_provision_model.joblib"

MIN_LABEL_COUNT = 5


# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

def load_dataset(filepath):

    texts = []
    labels = []

    with lzma.open(filepath, "rt", encoding="utf-8") as file:

        for line in file:

            record = json.loads(line)

            question = record.get("instruction", "")
            provisions = record.get("answer", [])

            if not isinstance(question, str) or not question.strip():
                continue

            if not isinstance(provisions, list):
                continue

            clean_provisions = [
                item.strip()
                for item in provisions
                if isinstance(item, str) and item.strip()
            ]

            texts.append(question.strip())
            labels.append(sorted(set(clean_provisions)))

    return texts, labels


print("\nLoading training dataset...")
X_train, y_train_raw = load_dataset(TRAIN_FILE)

print("Loading development dataset...")
X_dev, y_dev_raw = load_dataset(DEV_FILE)

print("Loading test dataset...")
X_test, y_test_raw = load_dataset(TEST_FILE)

print("\nDataset loaded successfully.")

print("Training questions:", len(X_train))
print("Development questions:", len(X_dev))
print("Testing questions:", len(X_test))


# --------------------------------------------------
# 3. SELECT FREQUENT LEGAL PROVISIONS
# --------------------------------------------------

label_counts = Counter(
    label
    for question_labels in y_train_raw
    for label in question_labels
)

selected_labels = sorted([
    label
    for label, count in label_counts.items()
    if count >= MIN_LABEL_COUNT
])

print("\nTotal unique training provisions:", len(label_counts))
print("Selected provisions:", len(selected_labels))
print("Minimum examples per provision:", MIN_LABEL_COUNT)


# --------------------------------------------------
# 4. ENCODE MULTIPLE OUTPUT LABELS
# --------------------------------------------------

encoder = MultiLabelBinarizer(classes=selected_labels)

encoder.fit([selected_labels])

def filter_labels(dataset_labels):

    return [
        [
            label
            for label in question_labels
            if label in encoder.classes_
        ]
        for question_labels in dataset_labels
    ]


y_train = encoder.transform(filter_labels(y_train_raw))
y_dev = encoder.transform(filter_labels(y_dev_raw))
y_test = encoder.transform(filter_labels(y_test_raw))


# Report how much of each split is covered by selected labels.

def report_coverage(name, raw_labels):

    total = sum(len(labels) for labels in raw_labels)

    covered = sum(
        1
        for labels in raw_labels
        for label in labels
        if label in encoder.classes_
    )

    percentage = (covered / total * 100) if total else 0

    print(
        f"{name} provision-label coverage: "
        f"{covered}/{total} ({percentage:.2f}%)"
    )


print("\nLABEL COVERAGE")
report_coverage("Train", y_train_raw)
report_coverage("Development", y_dev_raw)
report_coverage("Test", y_test_raw)


# --------------------------------------------------
# 5. BUILD MULTI-LABEL ML MODEL
# --------------------------------------------------

print("\nBuilding TF-IDF + Logistic Regression model...")

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=200000,
            sublinear_tf=True,
            strip_accents="unicode"
        )
    ),
    (
        "classifier",
        OneVsRestClassifier(
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="liblinear"
            ),
            n_jobs=-1
        )
    )
])


# --------------------------------------------------
# 6. TRAIN MODEL
# --------------------------------------------------

print("\nTraining started...")
print("This may take several minutes.")

model.fit(X_train, y_train)

print("\nTraining completed!")


# --------------------------------------------------
# 7. TUNE PREDICTION THRESHOLD USING DEV DATA
# --------------------------------------------------

print("\nFinding the best prediction threshold...")

dev_probabilities = model.predict_proba(X_dev)

best_threshold = 0.35
best_f1 = -1

for threshold in np.arange(0.10, 0.71, 0.025):

    predictions = (
        dev_probabilities >= threshold
    ).astype(int)

    score = f1_score(
        y_dev,
        predictions,
        average="micro",
        zero_division=0
    )

    if score > best_f1:

        best_f1 = score
        best_threshold = float(threshold)

print("\nBest threshold:", round(best_threshold, 3))
print("Development micro F1:", round(best_f1, 4))


# --------------------------------------------------
# 8. FINAL EVALUATION ON TEST DATA
# --------------------------------------------------

print("\nEvaluating on unseen test data...")

test_probabilities = model.predict_proba(X_test)

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

micro_f1 = f1_score(
    y_test,
    test_predictions,
    average="micro",
    zero_division=0
)

micro_precision = precision_score(
    y_test,
    test_predictions,
    average="micro",
    zero_division=0
)

micro_recall = recall_score(
    y_test,
    test_predictions,
    average="micro",
    zero_division=0
)

macro_f1 = f1_score(
    y_test,
    test_predictions,
    average="macro",
    zero_division=0
)

print("\n========== FINAL TEST RESULTS ==========")

print("Micro Precision:", round(micro_precision, 4))
print("Micro Recall:   ", round(micro_recall, 4))
print("Micro F1 Score: ", round(micro_f1, 4))
print("Macro F1 Score: ", round(macro_f1, 4))


# --------------------------------------------------
# 9. SAVE TRAINED MODEL
# --------------------------------------------------

bundle = {
    "model": model,
    "encoder": encoder,
    "threshold": best_threshold,
    "selected_labels": selected_labels
}

joblib.dump(bundle, MODEL_PATH)

print("\n========================================")
print("LEGAL PROVISION MODEL SAVED SUCCESSFULLY")
print("========================================")

print("Model location:", MODEL_PATH)
print("Number of provisions:", len(selected_labels))
print("Prediction threshold:", round(best_threshold, 3))
