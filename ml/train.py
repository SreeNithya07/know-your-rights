import os
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

DATA_PATH = os.path.join("data", "legal_intents.csv")
MODEL_PATH = os.path.join("artifacts", "incident_model.joblib")

df = pd.read_csv(DATA_PATH)
X = df["text"]
y = df["label"]

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

model.fit(X, y)

predictions = model.predict(X)
print(classification_report(y, predictions))

os.makedirs("artifacts", exist_ok=True)
joblib.dump(model, MODEL_PATH)

print("Model saved to:", MODEL_PATH)