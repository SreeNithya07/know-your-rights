
import lzma
import json
from pathlib import Path
from collections import Counter

BASE = Path("court-dataset") / "Layman-new-dataset"

TRAIN_FILE = BASE / "FT-Layman-train.jsonl.xz"
DEV_FILE = BASE / "FT-Layman-dev.jsonl.xz"
TEST_FILE = BASE / "FT-Layman-test.jsonl.xz"

train_counts = Counter()
dev_counts = Counter()
test_counts = Counter()

def read_labels(filepath, counter):
    with lzma.open(filepath, "rt", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            labels = record.get("answer", [])

            if isinstance(labels, list):
                for label in labels:
                    if isinstance(label, str) and label.strip():
                        counter[label.strip()] += 1

read_labels(TRAIN_FILE, train_counts)
read_labels(DEV_FILE, dev_counts)
read_labels(TEST_FILE, test_counts)

print("\n========== TRAIN LABEL DISTRIBUTION ==========")

for minimum in [1, 2, 3, 5, 10, 20, 50, 100]:
    count = sum(1 for value in train_counts.values()
                if value >= minimum)

    print(f"Labels appearing at least {minimum:3} times: {count}")

print("\n========== LABEL FREQUENCY GROUPS ==========")

groups = {
    "1 occurrence": 0,
    "2-4 occurrences": 0,
    "5-9 occurrences": 0,
    "10-19 occurrences": 0,
    "20-49 occurrences": 0,
    "50-99 occurrences": 0,
    "100+ occurrences": 0
}

for value in train_counts.values():
    if value == 1:
        groups["1 occurrence"] += 1
    elif value <= 4:
        groups["2-4 occurrences"] += 1
    elif value <= 9:
        groups["5-9 occurrences"] += 1
    elif value <= 19:
        groups["10-19 occurrences"] += 1
    elif value <= 49:
        groups["20-49 occurrences"] += 1
    elif value <= 99:
        groups["50-99 occurrences"] += 1
    else:
        groups["100+ occurrences"] += 1

for group, count in groups.items():
    print(f"{group:20}: {count}")

print("\n========== LABELS NOT SEEN IN TRAIN ==========")

unseen_dev = set(dev_counts) - set(train_counts)
unseen_test = set(test_counts) - set(train_counts)

print("Unseen in training but present in dev:", len(unseen_dev))
print("Unseen in training but present in test:", len(unseen_test))

print("\n========== TOP 20 LABELS ==========")

for label, count in train_counts.most_common(20):
    print(f"{count:5} | {label}")

print("\n========== RARE LABEL EXAMPLES ==========")

for label, count in sorted(train_counts.items(),
                           key=lambda item: item[1]):
    if count <= 2:
        print(f"{count} | {label}")
