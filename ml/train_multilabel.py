
import pandas as pd
import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "legal_issues.csv"
MODEL_PATH = BASE_DIR / "artifacts" / "legal_issue_model.joblib"

print("Loading legal training dataset...")

df = pd.read_csv(DATA_PATH).dropna()

# Separate user complaints and multiple legal labels
X = df["text"].astype(str)

labels = df["labels"].apply(
    lambda value: [
        item.strip()
        for item in value.split("|")
        if item.strip()
    ]
)

# Convert multiple labels into ML format
encoder = MultiLabelBinarizer()
y = encoder.fit_transform(labels)

print("\nTotal training examples:", len(X))
print("Total legal issue categories:", len(encoder.classes_))

print("\nLegal categories:")
for category in encoder.classes_:
    print("-", category)

# Create multi-label ML model
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True
    )),
    ("classifier", OneVsRestClassifier(
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        )
    ))
])

print("\nTraining multi-label legal model...")

model.fit(X, y)

# Training-set evaluation (not a test of unseen questions)
predictions = model.predict(X)

print("\nTRAINING DATA EVALUATION")
print(classification_report(
    y,
    predictions,
    target_names=encoder.classes_,
    zero_division=0
))

# Save model and label encoder together
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump({
    "model": model,
    "encoder": encoder,
    "threshold": 0.35
}, MODEL_PATH)

print("\nSUCCESS!")
print("Model saved to:", MODEL_PATH)
