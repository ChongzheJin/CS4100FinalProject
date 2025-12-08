"""
Training script for Agent 1 - Direct Coordinate Regression

Trains a CNN to predict latitude and longitude coordinates from street view images.
"""

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import sys
from datetime import datetime
import json
from tqdm import tqdm

# Add src to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

# UNCOMMENT TO DOWNLOAD MODEL
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

from data.datasets import GeoCSVDataset
from models.agent1_model import create_model
from utils.losses import HaversineLoss
from utils.coordinates import compute_normalization_params


def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_device():
    """Detect and return the best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps"), "Apple Silicon GPU"
    elif torch.cuda.is_available():
        return torch.device("cuda"), "NVIDIA GPU"
    else:
        return torch.device("cpu"), "CPU"


def should_use_pin_memory(device):
    """Check if pin_memory should be used for the device."""
    # MPS doesn't support pin_memory
    return device.type == "cuda"


def create_data_loaders(data_config, train_config, lat_min, lat_max, lon_min, lon_max, device):

    train_csv_path = ROOT / data_config['train_csv']
    val_csv_path = ROOT / data_config['val_csv']

    """Create train and validation data loaders."""
    train_dataset = GeoCSVDataset(
        csv_path=str(train_csv_path),
        img_size=data_config['img_size'],
        train=True,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        normalize=True
    )
    
    val_dataset = GeoCSVDataset(
        csv_path=str(val_csv_path),
        img_size=data_config['img_size'],
        train=False,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        normalize=True
    )
    
    use_pin_memory = should_use_pin_memory(device)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        num_workers=train_config.get('num_workers', 4),
        pin_memory=use_pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config['batch_size'],
        shuffle=False,
        num_workers=train_config.get('num_workers', 4),
        pin_memory=use_pin_memory
    )
    
    return train_loader, val_loader


def train_epoch(model, train_loader, criterion, optimizer, device, print_every_n=50):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    num_batches = len(train_loader)
    
    pbar = tqdm(enumerate(train_loader), total=num_batches, desc="Training")
    for batch_idx, (images, coords) in pbar:
        # Ensure images are float32 for MPS compatibility
        images = images.float().to(device)
        coords = coords.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, coords)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update running loss
        running_loss += loss.item()
        avg_loss = running_loss / (batch_idx + 1)
        
        # Update progress bar
        if (batch_idx + 1) % print_every_n == 0:
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    return running_loss / num_batches


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    num_batches = len(val_loader)
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating", leave=False)
        for images, coords in pbar:
            # Ensure images are float32 for MPS compatibility
            images = images.float().to(device)
            coords = coords.to(device)
            
            predictions = model(images)
            loss = criterion(predictions, coords)
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return running_loss / num_batches


def save_checkpoint(model, optimizer, scheduler, epoch, loss, checkpoint_dir, is_best=False):
    """Save model checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # Save latest checkpoint
    checkpoint_path = checkpoint_dir / 'latest_checkpoint.pt'
    torch.save(checkpoint, checkpoint_path)
    
    # Save best checkpoint
    if is_best:
        best_path = checkpoint_dir / 'best_checkpoint.pt'
        torch.save(checkpoint, best_path)
        print(f"  💾 Saved best checkpoint (loss: {loss:.4f})")
    
    # Save epoch checkpoint
    epoch_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
    torch.save(checkpoint, epoch_path)


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss']


def main():
    parser = argparse.ArgumentParser(description='Train Agent 1 - Coordinate Regression')
    parser.add_argument(
        '--data-config',
        type=str,
        default=ROOT / 'configs/data.yaml',
        help='Path to data configuration file'
    )
    parser.add_argument(
        '--train-config',
        type=str,
        default=ROOT / 'configs/train_agent1.yaml',
        help='Path to training configuration file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    args = parser.parse_args()
    
    # Load configurations
    print("=" * 60)
    print("Loading Configurations")
    print("=" * 60)
    data_config = load_config(args.data_config)
    train_config = load_config(args.train_config)
    
    print(f"Data config: {args.data_config}")
    print(f"Training config: {args.train_config}")
    
    # Detect device
    device, device_name = get_device()
    print(f"\nUsing device: {device_name} ({device})")
    
    # Compute normalization parameters from training data
    print("\n" + "=" * 60)
    print("Computing Normalization Parameters")
    print("=" * 60)
    train_csv_path = ROOT / data_config['train_csv']
    lat_min, lat_max, lon_min, lon_max = compute_normalization_params(str(train_csv_path))
    print(f"Latitude range: [{lat_min:.6f}, {lat_max:.6f}]")
    print(f"Longitude range: [{lon_min:.6f}, {lon_max:.6f}]")
    
    # Create data loaders
    print("\n" + "=" * 60)
    print("Creating Data Loaders")
    print("=" * 60)
    train_loader, val_loader = create_data_loaders(
        data_config, train_config['training'], lat_min, lat_max, lon_min, lon_max, device
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Create model
    print("\n" + "=" * 60)
    print("Creating Model")
    print("=" * 60)
    model_config = train_config['model']
    model = create_model(
        backbone=model_config['backbone'],
        pretrained=model_config['pretrained'],
        dropout=model_config['dropout'],
        freeze_backbone=model_config['freeze_backbone'],
        device=device
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create loss function
    criterion = HaversineLoss(
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        normalize=True
    ).to(device)

    optimizer_config = train_config['optimizer']
    lr = float(train_config['training']['learning_rate'])
    wd = float(train_config['training']['weight_decay'])
    opt_type = optimizer_config['type'].lower()

    if opt_type == 'adam':
        betas = optimizer_config.get('betas', [0.9, 0.999])
        betas = (float(betas[0]), float(betas[1]))
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=wd,
            betas=betas
        )
    elif opt_type == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=wd,
            momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_config['type']}")

    scheduler_config = train_config['scheduler']
    scheduler = None

    if scheduler_config['type'] == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=float(scheduler_config['factor']),
            patience=int(scheduler_config['patience']),
            min_lr=float(scheduler_config.get('min_lr', 0.0))
        )
    elif scheduler_config['type'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(train_config['training']['epochs']),
            eta_min=float(scheduler_config.get('min_lr', 0.0))
        )

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        start_epoch, _ = load_checkpoint(model, optimizer, scheduler, args.resume)
        print(f"Resuming from epoch {start_epoch}")
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    
    early_stopping_config = train_config['early_stopping']
    checkpoint_config = train_config['checkpoint']
    logging_config = train_config['logging']
    
    checkpoint_dir = ROOT / checkpoint_config['save_dir']
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    epochs_without_improvement = 0
    
    for epoch in range(start_epoch, train_config['training']['epochs']):
        print(f"\nEpoch {epoch + 1}/{train_config['training']['epochs']}")
        print("-" * 60)
        
        # Train
        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            print_every_n=logging_config['print_every_n_batches']
        )
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        if scheduler is not None:
            if scheduler_config['type'] == 'reduce_on_plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Track history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Print metrics
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Train Loss: {train_loss:.4f} km | Val Loss: {val_loss:.4f} km | LR: {current_lr:.2e}")
        
        # Check for improvement
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Save checkpoint
        save_every_n = checkpoint_config.get('save_every_n_epochs', 0)
        should_save_epoch = (save_every_n > 0) and ((epoch + 1) % save_every_n == 0)
        
        if checkpoint_config['save_best'] and is_best:
            save_checkpoint(model, optimizer, scheduler, epoch + 1, val_loss, checkpoint_dir, is_best=True)
        elif should_save_epoch:
            save_checkpoint(model, optimizer, scheduler, epoch + 1, val_loss, checkpoint_dir, is_best=False)
        
        # Early stopping
        if epochs_without_improvement >= early_stopping_config['patience']:
            print(f"\nEarly stopping triggered after {epochs_without_improvement} epochs without improvement")
            break
    
    # Save final checkpoint and history
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    save_checkpoint(model, optimizer, scheduler, epoch + 1, val_loss, checkpoint_dir, is_best=False)
    
    # Save training history
    history_path = checkpoint_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")
    
    print(f"\nBest validation loss: {best_val_loss:.4f} km")
    print(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()

