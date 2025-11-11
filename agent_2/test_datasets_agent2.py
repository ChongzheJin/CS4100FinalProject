import yaml
from datasets_agent2 import StreetViewGridDataset




#Checks that the CSV is read properly, the image is loaded/opened successfully, and each image has a valid label

# Load paths from YAML
with open("configs/data.yaml", "r") as f:
    cfg = yaml.safe_load(f)

csv_path = cfg["train_csv"]
print(f"Using train CSV: {csv_path}")

# Initialize dataset
dataset = StreetViewGridDataset(csv_path, img_size=cfg["img_size"])

print(f"Dataset loaded successfully.")
print(f"Total samples: {len(dataset)}")

# Inspect a few samples
for i in range(10):
    img, label = dataset[i]
    print(f"Sample {i}: Image shape {img.shape}, Grid label {label}")
