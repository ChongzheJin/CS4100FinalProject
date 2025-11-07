# Complete Dataset Setup Guide - From Kaggle to Training Ready

## Prerequisites

### 1. Install Kaggle API (if you haven't already)

**Option A: Using pip**
```bash
pip install kaggle
```

**Option B: If you get errors, try:**
```bash
pip install --upgrade kaggle
```

### 2. Get Your Kaggle API Credentials

1. Go to https://www.kaggle.com/ and log in
2. Click on your profile picture (top right)
3. Click "Settings"
4. Scroll down to "API" section
5. Click "Create New Token" - this downloads a file called `kaggle.json`

### 3. Place Your Kaggle Credentials

**On Mac/Linux:**
```bash
# Create the .kaggle directory if it doesn't exist
mkdir -p ~/.kaggle

# Move your downloaded kaggle.json file there
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json

# Set proper permissions (Kaggle requires this)
chmod 600 ~/.kaggle/kaggle.json
```

**On Windows:**
1. Press `Win + R`, type `%USERPROFILE%`, press Enter
2. Create a folder called `.kaggle` (with the dot at the start)
3. Move your `kaggle.json` file into that folder

---

## Step 2: Download the Dataset

### Method 1: Using Kaggle API (Recommended)

1. **Open your terminal/command prompt**
   - On Mac: Open Terminal
   - On Windows: Open Command Prompt or PowerShell

2. **Navigate to your project directory:**
   ```bash
   cd "/Users/armaandave/Library/CloudStorage/OneDrive-NortheasternUniversity/JuniorYear/semester1/cs4100/final/CS4100FinalProject"
   ```

3. **Create the directory where images will go:**
   ```bash
   mkdir -p data/raw/us_streetview/images
   ```

4. **Download the dataset:**
   ```bash
   kaggle datasets download -d eadfb5e8e6d6c14d0362fb8d0bb95640ceb7004a1e0803e2bfee97305aa39fb7 -p data/raw/us_streetview/
   ```

5. **Unzip the downloaded file:**
   ```bash
   cd data/raw/us_streetview/
   unzip *.zip -d images/
   cd ../../../
   ```

   **If `unzip` doesn't work, try:**
   ```bash
   # On Mac, you might need to install unzip first, or use:
   cd data/raw/us_streetview/
   python -m zipfile -e *.zip images/
   cd ../../../
   ```

### Method 2: Manual Download (If API doesn't work)

1. Go to: https://www.kaggle.com/datasets/eadfb5e8e6d6c14d0362fb8d0bb95640ceb7004a1e0803e2bfee97305aa39fb7
2. Click the "Download" button (you may need to accept terms)
3. Wait for download to complete (it's a large file)
4. Extract the ZIP file
5. Copy ALL the image files (they should be named like `35.0746,-106.6403.jpg`) into:
   ```
   CS4100FinalProject/data/raw/us_streetview/images/
   ```

---

## Step 3: Verify Your Images Are in the Right Place

**Check that images are there:**
```bash
# Count how many images you have
ls data/raw/us_streetview/images/*.jpg | wc -l

# Or on Windows PowerShell:
(Get-ChildItem data/raw/us_streetview/images/*.jpg).Count
```

**Check that filenames are correct:**
```bash
# List first 5 images to see the format
ls data/raw/us_streetview/images/*.jpg | head -5
```

You should see filenames like:
- `35.0746,-106.6403.jpg`
- `40.7128,-74.0060.jpg`
- `34.0522,-118.2437.jpg`

**If your images are in subfolders:**
If the Kaggle dataset has images in subfolders, you need to move them all to the `images/` folder:

```bash
# Find all images recursively and move them
find data/raw/us_streetview/images -name "*.jpg" -exec mv {} data/raw/us_streetview/images/ \;
```

---

## Step 4: Create the CSV File from Image Filenames

This script reads all your images and creates a CSV file listing them with their coordinates.

**Run the script:**
```bash
python src/dataio/data_preparation_script/build_csv_from_filenames.py
```

**What this does:**
- Scans all images in `data/raw/us_streetview/images/`
- Extracts latitude and longitude from each filename
- **Filters out Alaska and Hawaii** (only keeps continental US)
- Creates `data/processed/us_streetview/all_images.csv`

**Expected output:**
```
Processing images: 100%|████████| 20000/20000 [00:30<00:00, 666.67image/s]
Wrote 18500 rows -> /path/to/data/processed/us_streetview/all_images.csv
```

(Note: The number will be less than total images because Alaska/Hawaii are filtered out)

**Verify the CSV was created:**
```bash
# Check the file exists
ls -lh data/processed/us_streetview/all_images.csv

# Look at first few lines
head -5 data/processed/us_streetview/all_images.csv
```

You should see:
```
image_path,lat,lon
data/raw/us_streetview/images/35.0746,-106.6403.jpg,35.0746,-106.6403
data/raw/us_streetview/images/40.7128,-74.0060.jpg,40.7128,-74.0060
...
```

---

## Step 5: Split into Train/Validation/Test Sets

This splits your data into:
- **80% training** (for learning)
- **10% validation** (for checking during training)
- **10% test** (for final evaluation)

**Run the script:**
```bash
python src/dataio/data_preparation_script/split_csv_randomly.py
```

**Expected output:**
```
{'train.csv': 14800, 'val.csv': 1850, 'test.csv': 1850}
```

**Verify the splits were created:**
```bash
ls -lh data/processed/us_streetview/
```

You should see:
- `all_images.csv` (all images)
- `train.csv` (80% of images)
- `val.csv` (10% of images)
- `test.csv` (10% of images)

---

## Step 6: Verify Everything Works

**Test that the dataset can be loaded:**
```bash
python -c "from src.dataio.datasets import GeoCSVDataset; ds = GeoCSVDataset('data/processed/us_streetview/train.csv'); print(f'Dataset has {len(ds)} images'); img, coords = ds[0]; print(f'Image shape: {img.shape}, Coords: {coords}')"
```

**Expected output:**
```
Dataset has 14800 images
Image shape: torch.Size([3, 256, 256]), Coords: tensor([...])
```

---

## Troubleshooting

### Problem: "No module named 'kaggle'"
**Solution:** Install it: `pip install kaggle`

### Problem: "403 Forbidden" when downloading
**Solution:** Make sure your `kaggle.json` is in the right place with correct permissions

### Problem: "No images found"
**Solution:** 
- Check that images are in `data/raw/us_streetview/images/`
- Check that filenames match the pattern `lat,lon.jpg`
- Make sure you're in the project root directory

### Problem: "CSV has 0 rows"
**Solution:**
- Check that image filenames are exactly `lat,lon.jpg` (no spaces, no extra characters)
- Check that lat/lon are valid numbers
- Check that images are actually in the `images/` folder (not subfolders)

### Problem: Script runs but creates empty CSV
**Solution:**
- All your images might be Alaska/Hawaii (filtered out)
- Check a few image filenames to see their coordinates
- Try removing the Alaska/Hawaii filter temporarily to test

---

## Final Checklist

Before you start training, verify:

- [ ] Images are in `data/raw/us_streetview/images/`
- [ ] `all_images.csv` exists and has rows
- [ ] `train.csv`, `val.csv`, `test.csv` all exist
- [ ] All CSV files have the same number of rows as expected (80/10/10 split)
- [ ] Test dataset loading works (Step 6 above)

---

## Next Steps

Once everything is set up:
1. Your data is ready for Agent 1 training
2. The CSV files point to your images
3. You can now run `train_agent1.py` (once it's implemented)

---

## Quick Reference: Directory Structure

```
CS4100FinalProject/
├── data/
│   ├── raw/
│   │   └── us_streetview/
│   │       └── images/              ← YOUR IMAGES GO HERE
│   │           ├── 35.0746,-106.6403.jpg
│   │           ├── 40.7128,-74.0060.jpg
│   │           └── ... (all your images)
│   └── processed/
│       └── us_streetview/            ← CSVs GENERATED HERE
│           ├── all_images.csv        ← Created by build_csv_from_filenames.py
│           ├── train.csv             ← Created by split_csv_randomly.py
│           ├── val.csv               ← Created by split_csv_randomly.py
│           └── test.csv              ← Created by split_csv_randomly.py
```


