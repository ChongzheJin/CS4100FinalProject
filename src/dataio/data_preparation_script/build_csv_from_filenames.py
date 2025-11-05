import re, csv
from pathlib import Path
import tqdm

ROOT = Path(__file__).resolve().parents[3]
img_dir = ROOT / "data" / "raw" / "us_streetview" / "images"
out_csv = ROOT / "data" / "processed" / "us_streetview" / "all_images.csv"
out_csv.parent.mkdir(parents=True, exist_ok=True)

# Get lat/lon from filename
pat = re.compile(r"^(?P<lat>-?\d+(?:\.\d+)?),(?P<lon>-?\d+(?:\.\d+)?)\.(jpg|jpeg|png)$", re.IGNORECASE)

rows = []
for p in tqdm.tqdm(sorted(img_dir.rglob("*")), desc="Processing images", unit="image", total=len(list(img_dir.rglob("*")))):
    if p.suffix.lower() not in {".jpg",".jpeg",".png"}:
        continue
    m = pat.match(p.name)
    if not m:
        continue
    lat = float(m.group("lat"))
    lon = float(m.group("lon"))
    rel = p.resolve().relative_to(ROOT)
    rows.append([rel.as_posix(), lat, lon])

with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["image_path","lat","lon"])
    w.writerows(rows)

print(f"Wrote {len(rows)} rows -> {out_csv}")