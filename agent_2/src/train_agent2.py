import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets_agent2 import StreetViewGridDataset


# UNCOMMENT TO DOWNLOAD MODEL
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

# -----------------------------
# Constants & Config
# -----------------------------
CONFIG_PATH = "configs/data.yaml"
RESUME = False  # Set to True to resume from checkpoint
EPOCHS_1 = 10
EPOCHS_2 = 20
TOTAL_EPOCHS = EPOCHS_1 + EPOCHS_2
CORES = 4

def save_checkpoint(model, optimizer, epoch, history, save_dir, phase=""):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'phase': phase,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }
    checkpoint_path = os.path.join(save_dir, f'agent2_epoch_{epoch+1:03d}.pth')
    torch.save(checkpoint, checkpoint_path)
    torch.save(checkpoint, os.path.join(save_dir, 'latest.pth'))

def load_checkpoint(checkpoint_path, model, optimizer):
    """Load training checkpoint"""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'] + 1, checkpoint.get('history', {}), checkpoint.get('phase', 'phase1')
    return 0, {}, 'phase1'

def plot_metrics(history, save_path):
    """Plot training metrics with top-k accuracies"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Over Time')
    ax1.legend()
    ax1.grid(True)
    
    # Top-1 Accuracy
    ax2.plot(epochs, history['train_top1'], 'b-', label='Train')
    ax2.plot(epochs, history['val_top1'], 'r-', label='Val')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Top-1 Accuracy')
    ax2.set_title('Top-1 Accuracy (Exact Grid)')
    ax2.legend()
    ax2.grid(True)
    
    # Top-3 Accuracy
    ax3.plot(epochs, history['train_top3'], 'b-', label='Train')
    ax3.plot(epochs, history['val_top3'], 'r-', label='Val')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Top-3 Accuracy')
    ax3.set_title('Top-3 Accuracy (Nearby Grids)')
    ax3.legend()
    ax3.grid(True)
    
    # Top-5 Accuracy
    ax4.plot(epochs, history['train_top5'], 'b-', label='Train')
    ax4.plot(epochs, history['val_top5'], 'r-', label='Val')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Top-5 Accuracy')
    ax4.set_title('Top-5 Accuracy (General Region)')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# -----------------------------
# Helper Functions
# -----------------------------
def top_k_accuracy(outputs, labels, k):
    topk = outputs.topk(k, dim=1).indices
    correct = topk.eq(labels.view(-1, 1)).any(dim=1).sum().item()
    return correct / len(labels)

def run_epoch(model, loader, optimizer, criterion, train=True):
    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    correct1 = correct3 = correct5 = 0

    with torch.set_grad_enabled(train):
        for imgs, labels in tqdm(loader, desc="Train" if train else "Val"):
            imgs, labels = imgs.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

if __name__ == '__main__':

        # Load Config
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    train_csv = cfg["train_csv"]
    val_csv = cfg["val_csv"]
    img_size = cfg["img_size"]
    batch_size = cfg["batch_size"]
    num_classes = 49   # 7x7

    # -----------------------------
    # Data Augmentation
    # -----------------------------
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        # transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.Resize((img_size, img_size)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)
        ], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # -----------------------------
    # Dataset & Device Setup
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else 
                        "mps" if torch.backends.mps.is_available() else "cpu")
    print('using device:', device.type)

    train_dataset = StreetViewGridDataset(train_csv, img_size, transform=train_transform)
    val_dataset = StreetViewGridDataset(val_csv, img_size, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=CORES,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(CORES - 2, 0),
        persistent_workers=True,
        prefetch_factor=2
        )

    # -----------------------------
    # Compute Class Weights (GeoGuessrAI Style)
    # -----------------------------
    print("Computing class weights from dataset...")

    # access labels directly from the dataset's dataframe
    all_labels = train_dataset.df['grid_label'].values

    class_counts = torch.bincount(torch.tensor(all_labels), minlength=num_classes).float()
    
    # Calculate inverse frequency weights with square root smoothing (like GeoGuessrAI)
    total_images = len(all_labels)
    class_weights = torch.zeros(num_classes)
    
    for i in range(num_classes):
        if class_counts[i] > 0:
            # Inverse frequency weighting with square root smoothing
            # This matches the GeoGuessrAI formula: (total_images / (num_classes * count)) ** 0.5
            class_weights[i] = (total_images / (num_classes * class_counts[i])) ** 0.5
        else:
            class_weights[i] = 1.0  # Default weight for empty classes
    
    class_weights = class_weights.to(device)
    
    print(f'Class weights computed!')
    print(f'Weight range: [{class_weights.min():.3f}, {class_weights.max():.3f}]')
    print(f'Weight mean: {class_weights.mean():.3f}, std: {class_weights.std():.3f}')

    # -----------------------------
    # Model Setup
    # -----------------------------
    model = models.resnet50(weights="IMAGENET1K_V1")
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, num_classes),
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.fc.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    save_dir = "agent_2/saved_models"
    os.makedirs(save_dir, exist_ok=True)

    # -----------------------------
    # Initialize History
    # -----------------------------
    history = {
        'train_loss': [], 'val_loss': [],
        'train_top1': [], 'val_top1': [],
        'train_top3': [], 'val_top3': [],
        'train_top5': [], 'val_top5': []
    }

    start_epoch = 0
    current_phase = 'phase1'

    # Resume if needed
    if RESUME:
        checkpoint_path = os.path.join(save_dir, 'latest.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            start_epoch = checkpoint['epoch'] + 1
            history = checkpoint.get('history', {})
            current_phase = checkpoint.get('phase', 'phase1')
            
            # Load model state first
            model.load_state_dict(checkpoint['model_state_dict'])
            
            print(f"Loaded checkpoint from epoch {checkpoint['epoch'] + 1}, phase: {current_phase}")
    
    # -----------------------------
    # Training Loop
    # -----------------------------

    # Determine which phase we're in
    if start_epoch < EPOCHS_1:
        # Phase 1: Train Classifier Head
        print("\n=== Phase 1: Training last layer only ===\n")
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
        
        optimizer = optim.AdamW(model.fc.parameters(), lr=3e-4, weight_decay=0.01)
        
        for epoch in range(start_epoch, EPOCHS_1):
            print(f"\nEpoch {epoch+1}/{TOTAL_EPOCHS} (Phase 1 - Head Only)")
            
            train_loss, t1, t3, t5 = run_epoch(model, train_loader, optimizer, criterion, train=True)
            val_loss, v1, v3, v5 = run_epoch(model, val_loader, optimizer, criterion, train=False)

            scheduler.step(val_loss)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_top1'].append(t1)
            history['val_top1'].append(v1)
            history['train_top3'].append(t3)
            history['val_top3'].append(v3)
            history['train_top5'].append(t5)
            history['val_top5'].append(v5)
            
            print(f"Train Loss: {train_loss:.4f} | Top-1: {t1:.3f} | Top-3: {t3:.3f} | Top-5: {t5:.3f}")
            print(f"Val Loss:   {val_loss:.4f} | Top-1: {v1:.3f} | Top-3: {v3:.3f} | Top-5: {v5:.3f}")
            
            # Save checkpoint
            save_checkpoint(model, optimizer, epoch, history, save_dir, phase='phase1')
            
            # Plot every 2 epochs
            if (epoch + 1) % 2 == 0:
                plot_metrics(history, os.path.join(save_dir, 'training_plot.png'))
        
        start_epoch = EPOCHS_1

    # Phase 2: Fine-Tune Last Block
    if start_epoch >= EPOCHS_1:
        print("\n=== Phase 2: Fine-Tuning ResNet Block 4 ===\n")
        
        # Unfreeze layer4 and fc
        for name, param in model.named_parameters():
            if "layer4" in name or "fc" in name:
                param.requires_grad = True
        
        # Update optimizer with lower learning rate
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=0.001,
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=True
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=4, min_lr=1e-5
        )
        
        for epoch in range(start_epoch, TOTAL_EPOCHS):
            print(f"\nEpoch {epoch+1}/{TOTAL_EPOCHS} (Phase 2 - Fine-Tuning)")
            
            train_loss, t1, t3, t5 = run_epoch(model, train_loader, optimizer, criterion, train=True)
            val_loss, v1, v3, v5 = run_epoch(model, val_loader, optimizer, criterion, train=False)
            
            scheduler.step(val_loss)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_top1'].append(t1)
            history['val_top1'].append(v1)
            history['train_top3'].append(t3)
            history['val_top3'].append(v3)
            history['train_top5'].append(t5)
            history['val_top5'].append(v5)
            
            print(f"Train Loss: {train_loss:.4f} | Top-1: {t1:.3f} | Top-3: {t3:.3f} | Top-5: {t5:.3f}")
            print(f"Val Loss:   {val_loss:.4f} | Top-1: {v1:.3f} | Top-3: {v3:.3f} | Top-5: {v5:.3f}")
            
            # Save checkpoint
            save_checkpoint(model, optimizer, epoch, history, save_dir, phase='phase2')
            
            # Plot every 2 epochs
            if (epoch + 1) % 2 == 0:
                plot_metrics(history, os.path.join(save_dir, 'training_plot.png'))




    # Final plot
    plot_metrics(history, os.path.join(save_dir, 'final_training_plot.png'))
    print("\nTraining complete! Final plots saved.")