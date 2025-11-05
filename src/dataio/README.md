# DataIO – Data Preparation Guide

This folder contains all scripts related to **data preprocessing** and **dataset organization**  
for the GeoGuessr-AI project. It converts raw Street View images into CSV files that can be  
used by the training and evaluation pipelines of both agents on training, validation and testing.


Important: This script only works with the png, jpg and jpeg formats, and they have to be named as: 
`<latitude>,<longitude>.<extension>`
---

## Folder Structure
- Place your images in the root/data/raw/us_streetview/images folder
- The output will be placed in the root/data/processed/us_streetview folder

## Input Data
**Dataset source:**  
[Kaggle Street View (US)](https://www.kaggle.com/datasets/eadfb5e8e6d6c14d0362fb8d0bb95640ceb7004a1e0803e2bfee97305aa39fb7/data)

**Image format:**  
Each image file is named using its geographic coordinates only: `<latitude>,<longitude>.<extension>`

Examples:
- 35.0746,-106.6403.jpg
- 40.7128,-74.0060.jpg

## Step-by-Step Usage
### Generate the CSV listing (build step)
**Script:** `build_csv_from_filenames.py`

**Purpose:**  
Scans all images under `data/raw/us_streetview/images/`,  
extracts the latitude and longitude from each filename,  
and writes them into a structured CSV file.

**Run in PyCharm:**  
Right-click the script → “Run”.

**Output:**  
`data/processed/all_images.csv`

**CSV format:**
image_path, lat,lon

> *Note:* `image_path` is stored as a **relative path** so that the dataset is portable  
> across different machines.

---

### Split into train / val / test sets
**Script:** `split_random.py`

**Purpose:**
Randomly splits the dataset into 3 parts:
- **train.csv** – 80 % of samples  
- **val.csv** – 10 %  
- **test.csv** – 10 %

**Run in PyCharm:**
Right-click the script → “Run”.

**Output files:**
`data/processed/train.csv`
`data/processed/val.csv`
`data/processed/test.csv`

**CSV format:**
image_path, lat,lon

> *Note:* we are using `random.seed(2025)` to ensure reproducibility.

