"""
Stratified 85/15 train/val split across all 6 attack categories.
Output: data/train.jsonl, data/val.jsonl
"""
import json
import random
from pathlib import Path

DATASET_DIR = Path("dataset")
DATA_DIR    = Path("data")
TRAIN_RATIO = 0.85
SEED        = 42

CATEGORY_FILES = {
    "normal":           "normal_dataset.jsonl",
    "sql_injection":    "sql_injection_dataset.jsonl",
    "xss":              "xss_dataset.jsonl",
    "path_traversal":   "path_traversal_dataset.jsonl",
    "ssti":             "ssti_dataset.jsonl",
    "command_injection":"command_injection_dataset.jsonl",
}


def load_category(category: str, fname: str) -> list[dict]:
    records = []
    with open(DATASET_DIR / fname, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                obj["_category"] = category
                records.append(obj)
    return records


def save_jsonl(data: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            out = {k: v for k, v in item.items() if k != "_category"}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def main():
    random.seed(SEED)
    DATA_DIR.mkdir(exist_ok=True)

    train_all: list[dict] = []
    val_all:   list[dict] = []

    print("Per-category split:")
    for category, fname in CATEGORY_FILES.items():
        samples = load_category(category, fname)
        random.shuffle(samples)
        n_train = int(len(samples) * TRAIN_RATIO)
        train_all.extend(samples[:n_train])
        val_all.extend(samples[n_train:])
        print(f"  {category:22s}: {n_train:3d} train  {len(samples) - n_train:2d} val")

    random.shuffle(train_all)
    random.shuffle(val_all)

    save_jsonl(train_all, DATA_DIR / "train.jsonl")
    save_jsonl(val_all,   DATA_DIR / "val.jsonl")

    print(f"\nSaved → data/train.jsonl ({len(train_all)} samples)")
    print(f"        data/val.jsonl   ({len(val_all)} samples)")


if __name__ == "__main__":
    main()
