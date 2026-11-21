
import joblib
import numpy as np

MODEL_PATH = "ml/artifacts/legal_provision_model.joblib"

print("Loading trained legal model...")

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
encoder = bundle["encoder"]
threshold = bundle["threshold"]

print("Model loaded successfully!")
print("Prediction threshold:", threshold)


# Test questions representing different legal problems

test_questions = [
    "My husband beats me, threatens to throw me out of the house and takes all my salary.",

    "My ex boyfriend is blackmailing me with private pictures and threatening to share them online.",

    "My landlord refuses to return my security deposit and is threatening me.",

    "My parents are forcing me to marry someone against my wishes.",

    "I saw a teacher leaking examination question papers before the examination.",

    "Someone took money from me promising a job but never provided the job or returned my money."
]


for question in test_questions:

    print("\n" + "=" * 75)
    print("USER QUESTION:")
    print(question)

    probabilities = model.predict_proba([question])[0]

    # Get the 10 highest scoring provisions
    top_indices = np.argsort(probabilities)[::-1][:10]

    print("\nTOP 10 PREDICTED PROVISIONS:")

    for index in top_indices:

        provision = encoder.classes_[index]
        score = probabilities[index]

        status = "PREDICTED" if score >= threshold else "BELOW THRESHOLD"

        print(
            f"{score * 100:6.2f}% | {status:16} | {provision}"
        )

    selected = [
        encoder.classes_[i]
        for i, score in enumerate(probabilities)
        if score >= threshold
    ]

    print("\nPROVISIONS ABOVE THRESHOLD:")

    if selected:
        for provision in selected:
            print("-", provision)
    else:
        print("No provision passed the threshold.")
