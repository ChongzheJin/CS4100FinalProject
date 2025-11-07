#!/bin/bash
# Script to automatically find and process the Kaggle dataset download

echo "Looking for recently downloaded ZIP files..."
echo ""

# Find the most recently downloaded zip file
LATEST_ZIP=$(ls -t ~/Downloads/*.zip 2>/dev/null | head -1)

if [ -z "$LATEST_ZIP" ]; then
    echo "❌ No ZIP files found in Downloads folder"
    echo "Please download the dataset from Kaggle first"
    exit 1
fi

echo "Found: $LATEST_ZIP"
echo ""

# Create directory if it doesn't exist
mkdir -p data/raw/us_streetview/images

# Move the zip file
echo "Moving ZIP file to data/raw/us_streetview/..."
mv "$LATEST_ZIP" data/raw/us_streetview/

# Get the filename
ZIP_NAME=$(basename "$LATEST_ZIP")

echo "Extracting to data/raw/us_streetview/images/..."
cd data/raw/us_streetview/
python -m zipfile -e "$ZIP_NAME" images/

echo ""
echo "✅ Done! Images should now be in data/raw/us_streetview/images/"
echo ""
echo "Checking image count..."
IMAGE_COUNT=$(find images -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" 2>/dev/null | wc -l | tr -d ' ')
echo "Found $IMAGE_COUNT images"

cd ../../../

