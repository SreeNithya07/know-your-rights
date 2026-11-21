
import joblib
from pathlib import Path

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "artifacts"
    / "legal_issue_model.joblib"
)

print("Loading trained legal model...")

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
encoder = bundle["encoder"]
threshold = bundle["threshold"]

print("\nKNOW YOUR RIGHTS - LEGAL ISSUE PREDICTION")
print("Type 'exit' to stop.")

while True:

    question = input("\nDescribe your problem: ").strip()

    if question.lower() == "exit":
        break

    if not question:
        print("Please enter a legal problem.")
        continue

    probabilities = model.predict_proba([question])[0]

    results = []

    for index, score in enumerate(probabilities):

        if score >= threshold:
            results.append({
                "issue": encoder.classes_[index],
                "score": score
            })

    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    print("\nIDENTIFIED LEGAL ISSUES")
    print("-" * 40)

    if not results:
        print("No confident issue identified.")
        print("More training examples or manual review may be needed.")
    else:
        for result in results:
            print(
                f"{result['issue']} "
                f"(model score: {result['score']:.2f})"
            )
