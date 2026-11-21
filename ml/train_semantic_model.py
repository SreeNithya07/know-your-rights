
import os
import json
import lzma
import joblib
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score


# --------------------------------------------------
# PATHS AND SETTINGS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "court-dataset", "Layman-new-dataset")
ARTIFACT_DIR = os.path.join(BASE_DIR, "ml", "artifacts")

os.makedirs(ARTIFACT_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(DATA_DIR, "FT-Layman-train.jsonl.xz")
DEV_FILE = os.path.join(DATA_DIR, "FT-Layman-dev.jsonl.xz")
TEST_FILE = os.path.join(DATA_DIR, "FT-Layman-test.jsonl.xz")

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

OUTPUT_FILE = os.path.join(
    ARTIFACT_DIR,
    "legal_provision_semantic_model.joblib"
)


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

def load_dataset(filepath):

    questions = []
    labels = []

    print(f"\nLoading: {os.path.basename(filepath)}")

    with lzma.open(filepath, "rt", encoding="utf-8") as file:

        for line in file:

            if not line.strip():
                continue

            record = json.loads(line)

            question = record.get("instruction", "")
            answer = record.get("answer", [])

            if isinstance(answer, str):
                try:
                    answer = json.loads(answer)
                except json.JSONDecodeError:
                    answer = []

            if not isinstance(answer, list):
                continue

            provisions = [
                str(item).strip()
                for item in answer
                if isinstance(item, str) and item.strip()
            ]

            if question.strip() and provisions:
                questions.append(question.strip())
                labels.append(provisions)

    print("Questions loaded:", len(questions))

    return questions, labels


# --------------------------------------------------
# LOAD TRAIN, DEV AND TEST
# --------------------------------------------------

X_train, y_train = load_dataset(TRAIN_FILE)
X_dev, y_dev = load_dataset(DEV_FILE)
X_test, y_test = load_dataset(TEST_FILE)


# --------------------------------------------------
# SELECT LABELS
# Keep provisions appearing at least 5 times
# in the training dataset.
# --------------------------------------------------

from collections import Counter

label_counts = Counter(
    provision
    for row in y_train
    for provision in set(row)
)

selected_labels = sorted([
    label
    for label, count in label_counts.items()
    if count >= 5
])

mlb = MultiLabelBinarizer(classes=selected_labels)

Y_train = mlb.fit_transform(y_train)
Y_dev = mlb.transform(y_dev)
Y_test = mlb.transform(y_test)

print("\nNumber of legal provisions:", len(selected_labels))
print("Training matrix:", Y_train.shape)
print("Development matrix:", Y_dev.shape)
print("Testing matrix:", Y_test.shape)


# --------------------------------------------------
# CONVERT QUESTIONS INTO SEMANTIC EMBEDDINGS
# --------------------------------------------------

print("\nLoading sentence embedding model...")

embedding_model = SentenceTransformer(MODEL_NAME)

print("\nConverting training questions...")

E_train = embedding_model.encode(
    X_train,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)

print("\nConverting development questions...")

E_dev = embedding_model.encode(
    X_dev,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)

print("\nConverting testing questions...")

E_test = embedding_model.encode(
    X_test,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)

print("\nEmbedding shapes:")
print("Train:", E_train.shape)
print("Dev:", E_dev.shape)
print("Test:", E_test.shape)


# --------------------------------------------------
# TRAIN OUR OWN MULTI-LABEL CLASSIFIER
# --------------------------------------------------

print("\nTraining semantic legal provision classifier...")

classifier = OneVsRestClassifier(
    LogisticRegression(
        max_iter=1500,
        class_weight="balanced",
        solver="liblinear"
    ),
    n_jobs=-1
)

classifier.fit(E_train, Y_train)

print("\nTraining completed!")


# --------------------------------------------------
# FIND BEST THRESHOLD USING DEV DATA
# --------------------------------------------------

print("\nFinding best prediction threshold...")

dev_probabilities = classifier.predict_proba(E_dev)

best_threshold = 0.45
best_f1 = 0

for threshold in np.arange(0.10, 0.71, 0.025):

    predictions = (
        dev_probabilities >= threshold
    ).astype(int)

    score = f1_score(
        Y_dev,
        predictions,
        average="micro",
        zero_division=0
    )

    if score > best_f1:
        best_f1 = score
        best_threshold = float(threshold)

print("\nBest threshold:", best_threshold)
print("Development micro F1:", round(best_f1, 4))


# --------------------------------------------------
# FINAL TEST EVALUATION
# --------------------------------------------------

print("\nEvaluating on test dataset...")

test_probabilities = classifier.predict_proba(E_test)

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

precision = precision_score(
    Y_test,
    test_predictions,
    average="micro",
    zero_division=0
)

recall = recall_score(
    Y_test,
    test_predictions,
    average="micro",
    zero_division=0
)

micro_f1 = f1_score(
    Y_test,
    test_predictions,
    average="micro",
    zero_division=0
)

macro_f1 = f1_score(
    Y_test,
    test_predictions,
    average="macro",
    zero_division=0
)

print("\n===================================")
print("SEMANTIC MODEL TEST RESULTS")
print("===================================")

print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("Micro F1:", round(micro_f1, 4))
print("Macro F1:", round(macro_f1, 4))


# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

model_data = {
    "classifier": classifier,
    "mlb": mlb,
    "threshold": best_threshold,
    "embedding_model": MODEL_NAME
}

joblib.dump(model_data, OUTPUT_FILE)

print("\n===================================")
print("MODEL SAVED SUCCESSFULLY")
print("===================================")
print(OUTPUT_FILE)
