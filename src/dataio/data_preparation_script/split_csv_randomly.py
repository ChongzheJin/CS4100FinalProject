import csv, random
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[3]
in_csv = ROOT / "data" / "processed" / "us_streetview" / "all_images.csv"
out_dir = ROOT / "data" / "processed" / "us_streetview"
out_dir.mkdir(parents=True, exist_ok=True)

# Group images by grid cell
grid_to_images = defaultdict(list)
with open(in_csv, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Extract grid from path (e.g., "dataset/united_states/grid-3-4/...")
        path_parts = row['image_path'].split('/')
        grid_cell = None
        for part in path_parts:
            if part.startswith('grid-'):
                grid_cell = part
                break
        if grid_cell:
            grid_to_images[grid_cell].append(row)
        else:
            grid_to_images['unfit'].append(row)  # Handle unfit or other

# Show distribution
print(f"Total grids: {len(grid_to_images)}")
print(f"Images per grid:")
for grid, images in sorted(grid_to_images.items())[:10]:
    print(f"  {grid}: {len(images)} images")

# Split by GRIDS, not individual images
all_grids = list(grid_to_images.keys())
random.seed(2025)
random.shuffle(all_grids)

# Calculate split points for 80-10-10
n_grids = len(all_grids)
n_train_grids = int(n_grids * 0.8)  # 80% of grids for training
n_val_grids = int(n_grids * 0.1)    # 10% for validation

train_grids = all_grids[:n_train_grids]
val_grids = all_grids[n_train_grids:n_train_grids + n_val_grids]
test_grids = all_grids[n_train_grids + n_val_grids:]

print(f"\nGrid split:")
print(f"  Train: {len(train_grids)} grids ({len(train_grids)/n_grids*100:.1f}%)")
print(f"  Val: {len(val_grids)} grids ({len(val_grids)/n_grids*100:.1f}%)")
print(f"  Test: {len(test_grids)} grids ({len(test_grids)/n_grids*100:.1f}%)")

# Collect images for each split
splits = {
    "train.csv": [],
    "val.csv": [],
    "test.csv": []
}

for grid in train_grids:
    splits["train.csv"].extend(grid_to_images[grid])
for grid in val_grids:
    splits["val.csv"].extend(grid_to_images[grid])
for grid in test_grids:
    splits["test.csv"].extend(grid_to_images[grid])

# Calculate total images
total_images = sum(len(rows) for rows in splits.values())

# Shuffle images WITHIN each split (not across grids)
for split_name in splits:
    random.shuffle(splits[split_name])

# Write CSVs
for name, rows in splits.items():
    with open(out_dir / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_path","lat","lon"])
        w.writeheader()
        w.writerows(rows)

# Print statistics
print(f"\nImage distribution:")
for name, rows in splits.items():
    percentage = len(rows) / total_images * 100
    print(f"  {name}: {len(rows)} images ({percentage:.1f}%)")

# Verify no grid overlap
train_grid_set = set(train_grids)
val_grid_set = set(val_grids)
test_grid_set = set(test_grids)
print(f"\nOverlap check:")
print(f"  Train-Val overlap: {len(train_grid_set & val_grid_set)} grids (should be 0)")
print(f"  Train-Test overlap: {len(train_grid_set & test_grid_set)} grids (should be 0)")
print(f"  Val-Test overlap: {len(val_grid_set & test_grid_set)} grids (should be 0)")

# Show which grids are in each split (useful for debugging)
print(f"\nExample grids in each split:")
print(f"  Train grids (first 5): {train_grids[:5]}")
print(f"  Val grids: {val_grids[:5] if len(val_grids) >= 5 else val_grids}")
print(f"  Test grids: {test_grids[:5] if len(test_grids) >= 5 else test_grids}")