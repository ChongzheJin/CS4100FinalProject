"""
Test script for Haversine loss function.

Tests the loss function with known coordinates to verify correctness.
"""

import torch
import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "agent_1" / "src"))

from utils.losses import HaversineLoss, haversine_distance_km
from utils.coordinates import normalize_coordinates, denormalize_coordinates


def test_haversine_distance():
    """Test Haversine distance calculation with known coordinates."""
    print("=" * 60)
    print("Testing Haversine Distance Calculation")
    print("=" * 60)
    
    # Test 1: Same location (should be 0 km)
    print("\nTest 1: Same location")
    lat1 = torch.tensor([40.7128])  # New York City
    lon1 = torch.tensor([-74.0060])
    lat2 = torch.tensor([40.7128])
    lon2 = torch.tensor([-74.0060])
    
    distance = haversine_distance_km(lat1, lon1, lat2, lon2)
    print(f"  Location: NYC to NYC")
    print(f"  Expected: ~0 km")
    print(f"  Got: {distance.item():.4f} km")
    assert distance.item() < 0.01, "Same location should be ~0 km"
    print("  ✅ PASSED")
    
    # Test 2: NYC to Los Angeles (known distance ~3944 km)
    print("\nTest 2: NYC to Los Angeles")
    lat1 = torch.tensor([40.7128])  # NYC
    lon1 = torch.tensor([-74.0060])
    lat2 = torch.tensor([34.0522])  # LA
    lon2 = torch.tensor([-118.2437])
    
    distance = haversine_distance_km(lat1, lon1, lat2, lon2)
    print(f"  Location: NYC to LA")
    print(f"  Expected: ~3944 km")
    print(f"  Got: {distance.item():.2f} km")
    assert 3900 < distance.item() < 4000, f"Expected ~3944 km, got {distance.item()}"
    print("  ✅ PASSED")
    
    # Test 3: Batch of coordinates
    print("\nTest 3: Batch processing")
    lat1 = torch.tensor([40.7128, 34.0522])  # NYC, LA
    lon1 = torch.tensor([-74.0060, -118.2437])
    lat2 = torch.tensor([34.0522, 40.7128])  # LA, NYC
    lon2 = torch.tensor([-118.2437, -74.0060])
    
    distances = haversine_distance_km(lat1, lon1, lat2, lon2)
    print(f"  Batch size: 2")
    print(f"  Distances: {distances.tolist()}")
    assert len(distances) == 2, "Should return 2 distances"
    assert distances[0].item() > 3900, "NYC to LA should be ~3944 km"
    assert distances[1].item() > 3900, "LA to NYC should be ~3944 km"
    print("  ✅ PASSED")


def test_normalization():
    """Test coordinate normalization and denormalization."""
    print("\n" + "=" * 60)
    print("Testing Coordinate Normalization")
    print("=" * 60)
    
    # US bounds (continental)
    lat_min, lat_max = 24.455005, 49.049081
    lon_min, lon_max = -125.450687, -67.343249
    
    # Test coordinates (NYC)
    coords_deg = torch.tensor([[40.7128, -74.0060], [34.0522, -118.2437]])
    
    print(f"\nOriginal coordinates (degrees):")
    print(f"  {coords_deg.tolist()}")
    
    # Normalize
    coords_norm = normalize_coordinates(coords_deg, lat_min, lat_max, lon_min, lon_max)
    print(f"\nNormalized coordinates [-1, 1]:")
    print(f"  {coords_norm.tolist()}")
    
    # Check range
    assert torch.all(coords_norm >= -1) and torch.all(coords_norm <= 1), "Should be in [-1, 1]"
    print("  ✅ Normalized values are in [-1, 1] range")
    
    # Denormalize
    coords_denorm = denormalize_coordinates(coords_norm, lat_min, lat_max, lon_min, lon_max)
    print(f"\nDenormalized coordinates (degrees):")
    print(f"  {coords_denorm.tolist()}")
    
    # Check accuracy
    diff = torch.abs(coords_deg - coords_denorm)
    max_diff = diff.max().item()
    print(f"\nMaximum difference: {max_diff:.6f} degrees")
    assert max_diff < 0.001, "Denormalization should be accurate"
    print("  ✅ PASSED - Normalization round-trip successful")


def test_haversine_loss():
    """Test HaversineLoss class with normalized coordinates."""
    print("\n" + "=" * 60)
    print("Testing HaversineLoss Class")
    print("=" * 60)
    
    # US bounds
    lat_min, lat_max = 24.455005, 49.049081
    lon_min, lon_max = -125.450687, -67.343249
    
    # Create loss function
    loss_fn = HaversineLoss(
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        normalize=True
    )
    
    # Test 1: Perfect prediction (should be ~0 loss)
    print("\nTest 1: Perfect prediction")
    from utils.coordinates import normalize_coordinates
    
    true_coords_deg = torch.tensor([[40.7128, -74.0060]])  # NYC
    pred_coords_deg = torch.tensor([[40.7128, -74.0060]])  # Same
    
    true_coords_norm = normalize_coordinates(true_coords_deg, lat_min, lat_max, lon_min, lon_max)
    pred_coords_norm = normalize_coordinates(pred_coords_deg, lat_min, lat_max, lon_min, lon_max)
    
    loss = loss_fn(pred_coords_norm, true_coords_norm)
    print(f"  Expected: ~0 km")
    print(f"  Got: {loss.item():.4f} km")
    assert loss.item() < 0.01, "Perfect prediction should be ~0 km"
    print("  ✅ PASSED")
    
    # Test 2: NYC to LA prediction error
    print("\nTest 2: Large prediction error")
    true_coords_deg = torch.tensor([[40.7128, -74.0060]])  # NYC
    pred_coords_deg = torch.tensor([[34.0522, -118.2437]])  # LA (wrong!)
    
    true_coords_norm = normalize_coordinates(true_coords_deg, lat_min, lat_max, lon_min, lon_max)
    pred_coords_norm = normalize_coordinates(pred_coords_deg, lat_min, lat_max, lon_min, lon_max)
    
    loss = loss_fn(pred_coords_norm, true_coords_norm)
    print(f"  Expected: ~3944 km")
    print(f"  Got: {loss.item():.2f} km")
    assert 3900 < loss.item() < 4000, f"Expected ~3944 km, got {loss.item()}"
    print("  ✅ PASSED")
    
    # Test 3: Batch processing
    print("\nTest 3: Batch processing")
    true_coords_deg = torch.tensor([
        [40.7128, -74.0060],  # NYC
        [34.0522, -118.2437]  # LA
    ])
    pred_coords_deg = torch.tensor([
        [40.7128, -74.0060],  # Correct
        [40.7128, -74.0060]   # Wrong (predicted NYC instead of LA)
    ])
    
    true_coords_norm = normalize_coordinates(true_coords_deg, lat_min, lat_max, lon_min, lon_max)
    pred_coords_norm = normalize_coordinates(pred_coords_deg, lat_min, lat_max, lon_min, lon_max)
    
    loss = loss_fn(pred_coords_norm, true_coords_norm)
    print(f"  Batch size: 2")
    print(f"  Loss (mean): {loss.item():.2f} km")
    # First is correct (0 km), second is wrong (~3944 km), mean should be ~1972 km
    assert 1900 < loss.item() < 2000, f"Expected ~1972 km, got {loss.item()}"
    print("  ✅ PASSED")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("HAVERSINE LOSS FUNCTION TESTS")
    print("=" * 60)
    
    try:
        test_haversine_distance()
        test_normalization()
        test_haversine_loss()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nThe loss function is working correctly.")
        print("You can now use it in your training script.")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

