
import lzma
import json
from pathlib import Path

ROOT = Path("court-dataset")

datasets = [
    ROOT / "Layman-new-dataset" / "FT-Layman-train.jsonl.xz",
    ROOT / "Layman-subset-data" / "FT-LMsub-train.jsonl.xz",
    ROOT / "synthetic-layman-dataset" / "FT-SyntheticLM-train.jsonl.xz",
    ROOT / "FT-Court-train.jsonl.xz"
]

for filepath in datasets:

    print("\n" + "=" * 75)
    print("DATASET:", filepath)
    print("=" * 75)

    total = 0

    with lzma.open(filepath, "rt", encoding="utf-8") as file:

        for line in file:
            total += 1

            if total <= 3:
                record = json.loads(line)

                print("\nRECORD", total)
                print("Fields:", list(record.keys()))

                for key, value in record.items():
                    print(f"\n{key}:")
                    print(str(value)[:1500])

    print("\nTOTAL RECORDS:", total)
