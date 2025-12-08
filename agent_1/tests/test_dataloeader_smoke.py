import torch
from pathlib import Path
from torch.utils.data import DataLoader
import sys

ROOT = Path(__file__).resolve().parents[2]  # Goes to CS4100FinalProject
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "agent_1" / "src"))  # Add agent_1/src to path

from data.datasets import GeoCSVDataset
from utils.coordinates import compute_normalization_params  # Now this will work

TRAIN_CSV = ROOT / "data" / "processed" / "us_streetview" / "train.csv"

def test_smoke_dataloader():
    # Compute normalization parameters
    lat_min, lat_max, lon_min, lon_max = compute_normalization_params(str(TRAIN_CSV))
    
    # Create dataset with normalization
    ds = GeoCSVDataset(
        str(TRAIN_CSV),
        img_size=256,
        train=True,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        normalize=True
    )
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)

    x, y = next(iter(dl))

    # Assert input images have 4 dimensions and 3 channels
    assert x.ndim == 4 and x.shape[1] == 3
    # Assert labels have 2 elements per sample (latitude and longitude)
    assert y.shape[-1] == 2
    
    # Assert coordinates are normalized to [-1, 1] range
    assert torch.all(y >= -1) and torch.all(y <= 1), \
        f"Coordinates should be in [-1, 1], got min={y.min().item():.4f}, max={y.max().item():.4f}"
    
    print("✓ DataLoader smoke test passed!")
    print(f"  Batch shape: {x.shape}")
    print(f"  Labels shape: {y.shape}")
    print(f"  Coordinate range: [{y.min().item():.4f}, {y.max().item():.4f}]")

if __name__ == "__main__":
    test_smoke_dataloader()