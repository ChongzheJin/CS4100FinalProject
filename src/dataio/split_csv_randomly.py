import csv, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
in_csv = ROOT / "data" / "processed" / "all_images.csv"
out_dir = ROOT / "data" / "processed"
out_dir.mkdir(parents=True, exist_ok=True)

with open(in_csv, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

random.seed(2025)
random.shuffle(rows)
n = len(rows)
n_train = int(n*0.8)
n_eval_test = int(n*0.1)

splits = {
    "train.csv": rows[:n_train],
    "val.csv":   rows[n_train : n_train + n_eval_test],
    "test.csv":  rows[n_train + n_eval_test:]
}

for name, part in splits.items():
    with open(out_dir / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_path","lat","lon"])
        w.writeheader()
        w.writerows(part)

print({k: len(v) for k,v in splits.items()})