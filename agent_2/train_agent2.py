import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from tqdm import tqdm
import pandas as pd
from datasets_agent2 import StreetViewGridDataset


# -----------------------------
# Load Config
# -----------------------------
CONFIG_PATH = "configs/data.yaml"
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

train_csv = cfg["train_csv"]
val_csv = cfg["val_csv"]
img_size = cfg["img_size"]
batch_size = cfg["batch_size"]

# Number of grids (must match grid_mapper)
num_classes = 49   # 7x7



# -----------------------------
# Data Augmentation
# -----------------------------



# Image Normalization (ResNet standard)
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.3
    ),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

val_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])




# -----------------------------
# Dataset Initialization
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else 
                      "mps" if torch.backends.mps.is_available() else "cpu")

train_dataset = StreetViewGridDataset(train_csv, img_size, transform=train_transform)
val_dataset   = StreetViewGridDataset(val_csv,   img_size, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)



# -----------------------------
# Compute Class Weights from Dataset
# -----------------------------
print("Computing class weights from dataset...")

all_labels = [train_dataset[i][1] for i in range(len(train_dataset))]

# Count occurrences for each grid (0 to num_classes-1)
class_counts = torch.bincount(torch.tensor(all_labels), minlength=num_classes)

# Inverse-frequency weighting
class_weights = 1.0 / (class_counts + 1e-6)
class_weights = class_weights / class_weights.sum()

class_weights = class_weights.to(device)

print("Class weights computed:", class_weights)


# -----------------------------
# Model Setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else 
                      "mps" if torch.backends.mps.is_available() else "cpu")

model = models.resnet50(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# -----------------------------
# Accuracy Metrics
# -----------------------------
def top_k_accuracy(outputs, labels, k):
    topk = outputs.topk(k, dim=1).indices
    correct = topk.eq(labels.view(-1, 1)).any(dim=1).sum().item()
    return correct / len(labels)


# -----------------------------
# Training Function
# -----------------------------
def run_epoch(model, loader, train=True):
    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    correct1 = correct3 = correct5 = 0

    for imgs, labels in tqdm(loader, desc="Train" if train else "Val"):
        imgs, labels = imgs.to(device), labels.to(device)

        if train:
            optimizer.zero_grad()

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        if train:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        correct1 += top_k_accuracy(outputs, labels, 1) * imgs.size(0)
        correct3 += top_k_accuracy(outputs, labels, 3) * imgs.size(0)
        correct5 += top_k_accuracy(outputs, labels, 5) * imgs.size(0)

    avg_loss = running_loss / len(loader.dataset)
    acc1 = correct1 / len(loader.dataset)
    acc3 = correct3 / len(loader.dataset)
    acc5 = correct5 / len(loader.dataset)

    return avg_loss, acc1, acc3, acc5


# -----------------------------
# Phase 1: Train Classifier Head
# -----------------------------
print("\n=== Phase 1: Training last layer only ===\n")
for param in model.parameters():
    param.requires_grad = False

for param in model.fc.parameters():
    param.requires_grad = True

EPOCHS_1 = 5
for epoch in range(EPOCHS_1):
    print(f"\nEpoch {epoch+1}/{EPOCHS_1} (Head Only)")
    train_loss, t1, t3, t5 = run_epoch(model, train_loader, train=True)
    val_loss, v1, v3, v5 = run_epoch(model, val_loader, train=False)

    print(f"Train Loss: {train_loss:.4f} | Top-1: {t1:.3f} | Top-3: {t3:.3f} | Top-5: {t5:.3f}")
    print(f"Val Loss:   {val_loss:.4f} | Top-1: {v1:.3f} | Top-3: {v3:.3f} | Top-5: {v5:.3f}")


# -----------------------------
# Phase 2: Fine-Tune Last Block
# -----------------------------
print("\n=== Phase 2: Fine-Tuning ResNet Block 4 ===\n")

for name, param in model.named_parameters():
    if "layer4" in name or "fc" in name:
        param.requires_grad = True

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

EPOCHS_2 = 10
save_dir = "agent_2/saved_models"
os.makedirs(save_dir, exist_ok=True)

for epoch in range(EPOCHS_2):
    print(f"\nEpoch {epoch+1}/{EPOCHS_2} (Fine-Tuning)")
    train_loss, t1, t3, t5 = run_epoch(model, train_loader, train=True)
    val_loss, v1, v3, v5 = run_epoch(model, val_loader, train=False)

    print(f"Train Loss: {train_loss:.4f} | Top-1: {t1:.3f} | Top-3: {t3:.3f} | Top-5: {t5:.3f}")
    print(f"Val Loss:   {val_loss:.4f} | Top-1: {v1:.3f} | Top-3: {v3:.3f} | Top-5: {v5:.3f}")

    torch.save(model.state_dict(), os.path.join(save_dir, f"agent2_finetune_epoch{epoch+1}.pth"))

print("Training complete") 
