
import lzma
import json
from pathlib import Path
from collections import Counter

BASE = Path("court-dataset") / "Layman-new-dataset"

FILES = [
    "FT-Layman-train.jsonl.xz",
    "FT-Layman-dev.jsonl.xz",
    "FT-Layman-test.jsonl.xz"
]

for filename in FILES:

    filepath = BASE / filename

    print("\n" + "=" * 65)
    print("ANALYZING:", filename)
    print("=" * 65)

    total_records = 0
    total_provisions = 0
    provision_counts = Counter()
    answer_lengths = Counter()

    with lzma.open(filepath, "rt", encoding="utf-8") as file:

        for line in file:

            record = json.loads(line)

            question = record.get("instruction", "")
            answer = record.get("answer", [])

            if not isinstance(answer, list):
                continue

            total_records += 1
            total_provisions += len(answer)

            answer_lengths[len(answer)] += 1

            for provision in answer:
                provision_counts[provision.strip()] += 1

    print("Total questions:", total_records)
    print("Total legal provisions:", total_provisions)
    print("Unique legal provisions:", len(provision_counts))

    print("\nTOP 30 MOST COMMON LEGAL PROVISIONS:")

    for provision, count in provision_counts.most_common(30):
        print(f"{count:5} | {provision}")

    print("\nNUMBER OF PROVISIONS PER QUESTION:")

    for count, frequency in sorted(answer_lengths.items()):
        print(f"{count} provisions: {frequency} questions")
