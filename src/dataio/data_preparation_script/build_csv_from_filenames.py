import re, csv
from pathlib import Path
import tqdm

ROOT = Path(__file__).resolve().parents[3]
img_dir = ROOT / "dataset" / "united_states" 
out_csv = ROOT / "data" / "processed" / "us_streetview" / "all_images.csv"
out_csv.parent.mkdir(parents=True, exist_ok=True)

# Patterns
grid_dir_pattern = re.compile(r'^grid-\d+-\d+$')
lat_lon_pattern = re.compile(r"^(?P<lat>-?\d+(?:\.\d+)?),(?P<lon>-?\d+(?:\.\d+)?)\.(jpg|jpeg|png)$", re.IGNORECASE)

# First collect all valid image files
valid_files = []
for dir_path in img_dir.iterdir():
    if dir_path.is_dir():
        # Only process grid-X-Y directories and optionally 'unfit'
        if grid_dir_pattern.match(dir_path.name) or dir_path.name == "unfit":
            # Add all jpg files from this directory
            valid_files.extend(dir_path.glob("*.jpg"))
            valid_files.extend(dir_path.glob("*.JPG"))

print(f"Found {len(valid_files)} image files in grid directories")

# Process files with progress bar
rows = []
for p in tqdm.tqdm(sorted(valid_files), desc="Processing images", unit="image"):
    m = lat_lon_pattern.match(p.name)
    if not m:
        continue
    lat = float(m.group("lat"))
    lon = float(m.group("lon"))
    rel = p.resolve().relative_to(ROOT)
    rows.append([rel.as_posix(), lat, lon])

# Write CSV
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["image_path","lat","lon"])
    w.writerows(rows)

print(f"Wrote {len(rows)} rows -> {out_csv}")

# Verify the counts
import pandas as pd
df = pd.read_csv(out_csv)
print(f"\nVerification:")
print(f"CSV contains {len(df)} rows")
print(f"Expected ~35,320 based on your grid counts")