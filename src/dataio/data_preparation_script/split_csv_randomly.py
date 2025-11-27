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
        # Skip unfit or any non-grid images

# Filter out grids with less than 250 images
MIN_IMAGES = 250
filtered_grids = {grid: images for grid, images in grid_to_images.items() 
                  if len(images) >= MIN_IMAGES}

print(f"Total grids: {len(grid_to_images)}")
print(f"Grids with >= {MIN_IMAGES} images: {len(filtered_grids)}")
print(f"\nExcluded grids (< {MIN_IMAGES} images):")
for grid, images in sorted(grid_to_images.items()):
    if len(images) < MIN_IMAGES:
        print(f"  {grid}: {len(images)} images (excluded)")

print(f"\nIncluded grids (>= {MIN_IMAGES} images):")
for grid, images in sorted(filtered_grids.items()):
    print(f"  {grid}: {len(images)} images")

# Initialize splits
splits = {
    "train.csv": [],
    "val.csv": [],
    "test.csv": []
}

# Split EACH grid's images 80-10-10 (only for grids with enough images)
random.seed(2025)
for grid, images in filtered_grids.items():
    # Shuffle images within this grid
    random.shuffle(images)
    
    n = len(images)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    
    # Split this grid's images
    splits["train.csv"].extend(images[:n_train])
    splits["val.csv"].extend(images[n_train:n_train + n_val])
    splits["test.csv"].extend(images[n_train + n_val:])

# Shuffle each split to mix images from different grids
for split_name in splits:
    random.shuffle(splits[split_name])

# Write CSVs
for name, rows in splits.items():
    with open(out_dir / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_path","lat","lon"])
        w.writeheader()
        w.writerows(rows)

# Print statistics
total_images = sum(len(rows) for rows in splits.values())
print(f"\nImage distribution:")
for name, rows in splits.items():
    percentage = len(rows) / total_images * 100
    print(f"  {name}: {len(rows)} images ({percentage:.1f}%)")

# Verify each grid is represented in all splits
print(f"\nGrid representation check:")
for split_name, split_rows in splits.items():
    grids_in_split = set()
    for row in split_rows:
        path_parts = row['image_path'].split('/')
        for part in path_parts:
            if part.startswith('grid-'):
                grids_in_split.add(part)
                break
    print(f"  {split_name}: {len(grids_in_split)} unique grids")

print(f"\nFinal number of classes for model: {len(filtered_grids)}")