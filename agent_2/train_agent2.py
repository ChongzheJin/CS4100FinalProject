
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm
from datasets_agent2 import StreetViewGridDataset


# Load Config
CONFIG_PATH = "configs/data.yaml"
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

train_csv = cfg["train_csv"]
val_csv = cfg["val_csv"]
img_size = cfg["img_size"]
batch_size = cfg["batch_size"]


# Dataset & DataLoader
train_dataset = StreetViewGridDataset(train_csv, img_size)
val_dataset = StreetViewGridDataset(val_csv, img_size)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

num_classes = 49


# Model Setup
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, num_classes)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# Training Loop
EPOCHS = 10
save_dir = "agent_2/saved_models"
os.makedirs(save_dir, exist_ok=True)

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * imgs.size(0)

    avg_train_loss = train_loss / len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss, correct = 0.0, 0
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader.dataset)
    val_acc = correct / len(val_loader.dataset)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # Save model checkpoint
    torch.save(model.state_dict(), os.path.join(save_dir, f"agent2_epoch{epoch+1:02d}.pth"))

print("Training complete. Model saved to agent2/saved_models/")
