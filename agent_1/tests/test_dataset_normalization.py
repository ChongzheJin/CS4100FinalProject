"""
Test script for dataset normalization.

Verifies that the dataset correctly normalizes coordinates.
"""

import sys
from pathlib import Path
import torch
import pandas as pd

# Add src to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data.datasets import GeoCSVDataset
sys.path.insert(0, str(ROOT / "agent_1" / "src"))
from utils.coordinates import compute_normalization_params, denormalize_coordinates



def test_dataset_normalization():
    """Test that dataset correctly normalizes coordinates."""
    print("=" * 60)
    print("Testing Dataset Normalization")
    print("=" * 60)
    
    # Path to training CSV
    train_csv = ROOT / "data" / "processed" / "us_streetview" / "train.csv"
    
    if not train_csv.exists():
        print(f"\n❌ Training CSV not found at: {train_csv}")
        print("Please run the data preparation scripts first.")
        return False
    
    print(f"\nLoading dataset from: {train_csv}")
    
    # Compute normalization parameters from training data
    print("\nComputing normalization parameters...")
    lat_min, lat_max, lon_min, lon_max = compute_normalization_params(str(train_csv))
    print(f"  Latitude range: [{lat_min:.6f}, {lat_max:.6f}]")
    print(f"  Longitude range: [{lon_min:.6f}, {lon_max:.6f}]")
    
    # Create dataset with normalization
    print("\nCreating dataset with normalization...")
    dataset = GeoCSVDataset(
        csv_path=str(train_csv),
        img_size=256,
        train=True,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        normalize=True
    )
    
    print(f"  Dataset size: {len(dataset)} images")
    
    # Test 1: Check that coordinates are normalized
    print("\nTest 1: Checking coordinate normalization...")
    img, coords = dataset[0]
    print(f"  Sample coordinates: {coords.tolist()}")
    
    # Check range
    assert torch.all(coords >= -1) and torch.all(coords <= 1), \
        f"Coordinates should be in [-1, 1], got {coords.tolist()}"
    print("  ✅ Coordinates are in [-1, 1] range")
    
    # Test 2: Verify normalization round-trip
    print("\nTest 2: Verifying normalization round-trip...")
    # Get original coordinates from CSV
    import pandas as pd
    df = pd.read_csv(train_csv)
    original_lat = float(df.iloc[0]["lat"])
    original_lon = float(df.iloc[0]["lon"])
    
    # Denormalize the normalized coordinates
    coords_batch = coords.unsqueeze(0)  # Add batch dimension
    denorm_coords = denormalize_coordinates(
        coords_batch,
        lat_min,
        lat_max,
        lon_min,
        lon_max
    )
    denorm_lat = denorm_coords[0, 0].item()
    denorm_lon = denorm_coords[0, 1].item()
    
    print(f"  Original: ({original_lat:.6f}, {original_lon:.6f})")
    print(f"  Denormalized: ({denorm_lat:.6f}, {denorm_lon:.6f})")
    
    diff_lat = abs(original_lat - denorm_lat)
    diff_lon = abs(original_lon - denorm_lon)
    max_diff = max(diff_lat, diff_lon)
    
    print(f"  Maximum difference: {max_diff:.6f} degrees")
    assert max_diff < 0.001, f"Denormalization error too large: {max_diff}"
    print("  ✅ Normalization round-trip successful")
    
    # Test 3: Check multiple samples
    print("\nTest 3: Checking multiple samples...")
    all_in_range = True
    for i in range(min(10, len(dataset))):
        _, coords = dataset[i]
        if not (torch.all(coords >= -1) and torch.all(coords <= 1)):
            all_in_range = False
            print(f"  ❌ Sample {i} out of range: {coords.tolist()}")
    
    assert all_in_range, "Some samples are not normalized correctly"
    print(f"  ✅ All {min(10, len(dataset))} samples are normalized correctly")
    
    # Test 4: Test without normalization (for comparison)
    print("\nTest 4: Testing without normalization...")
    dataset_no_norm = GeoCSVDataset(
        csv_path=str(train_csv),
        img_size=256,
        train=True,
        normalize=False
    )
    _, coords_no_norm = dataset_no_norm[0]
    print(f"  Coordinates (no normalization): {coords_no_norm.tolist()}")
    assert torch.all(coords_no_norm >= -180) and torch.all(coords_no_norm <= 180), \
        "Unnormalized coordinates should be in degrees"
    print("  ✅ Non-normalized coordinates are in degrees")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nDataset normalization is working correctly.")
    print("You can now use the dataset in your training script.")
    
    return True


if __name__ == "__main__":
    try:
        import torch
        success = test_dataset_normalization()
        if not success:
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

