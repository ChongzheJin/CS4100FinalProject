"""
Test script for Agent 1 model architecture.

Verifies that the model:
- Loads correctly
- Processes images correctly
- Outputs normalized coordinates in correct shape/range
"""

import sys
import torch
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "agent_1" / "src"))

from models.agent1_model import GeoLocationRegressor, create_model


def test_model_creation():
    """Test model creation and basic properties."""
    print("=" * 60)
    print("Testing Model Creation")
    print("=" * 60)
    
    # Test 1: Create model
    print("\nTest 1: Creating model...")
    model = create_model(
        backbone="efficientnet_b0",
        pretrained=True,
        dropout=0.3,
        freeze_backbone=False
    )
    print("  ✅ Model created successfully")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Test 2: Model structure
    print("\nTest 2: Checking model structure...")
    assert hasattr(model, 'backbone'), "Model should have backbone"
    assert hasattr(model, 'regressor'), "Model should have regressor"
    print("  ✅ Model structure correct")
    
    return model


def test_forward_pass():
    """Test forward pass with dummy data."""
    print("\n" + "=" * 60)
    print("Testing Forward Pass")
    print("=" * 60)
    
    model = create_model(backbone="efficientnet_b0", pretrained=True)
    model.eval()  # Set to evaluation mode
    
    # Test 1: Single image
    print("\nTest 1: Single image forward pass...")
    batch_size = 1
    img_size = 256
    x = torch.randn(batch_size, 3, img_size, img_size)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output values: {output.tolist()}")
    
    assert output.shape == (batch_size, 2), \
        f"Expected shape ({batch_size}, 2), got {output.shape}"
    assert torch.all(output >= -1) and torch.all(output <= 1), \
        f"Output should be in [-1, 1], got min={output.min().item():.4f}, max={output.max().item():.4f}"
    print("  ✅ Single image forward pass successful")
    
    # Test 2: Batch of images
    print("\nTest 2: Batch forward pass...")
    batch_size = 8
    x = torch.randn(batch_size, 3, img_size, img_size)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    
    assert output.shape == (batch_size, 2), \
        f"Expected shape ({batch_size}, 2), got {output.shape}"
    assert torch.all(output >= -1) and torch.all(output <= 1), \
        "All outputs should be in [-1, 1]"
    print("  ✅ Batch forward pass successful")
    
    # Test 3: Different image sizes (should still work due to adaptive pooling)
    print("\nTest 3: Different image sizes...")
    sizes = [224, 256, 299]
    for size in sizes:
        x = torch.randn(1, 3, size, size)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (1, 2), f"Failed for size {size}"
    print(f"  ✅ Works with sizes: {sizes}")


def test_device_placement():
    """Test model on different devices."""
    print("\n" + "=" * 60)
    print("Testing Device Placement")
    print("=" * 60)
    
    # Detect available device
    if torch.backends.mps.is_available():
        device = "mps"
        device_name = "Apple Silicon GPU"
    elif torch.cuda.is_available():
        device = "cuda"
        device_name = "NVIDIA GPU"
    else:
        device = "cpu"
        device_name = "CPU"
    
    print(f"\nUsing device: {device_name} ({device})")
    
    model = create_model(backbone="efficientnet_b0", pretrained=True, device=device)
    x = torch.randn(2, 3, 256, 256).to(device)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"  Input device: {x.device}")
    print(f"  Output device: {output.device}")
    print(f"  Output shape: {output.shape}")
    
    assert output.device.type == device, f"Output should be on {device}"
    print(f"  ✅ Model works on {device_name}")


def test_backbone_features():
    """Test feature extraction."""
    print("\n" + "=" * 60)
    print("Testing Feature Extraction")
    print("=" * 60)
    
    model = create_model(backbone="efficientnet_b0", pretrained=True)
    model.eval()
    
    x = torch.randn(1, 3, 256, 256)
    
    with torch.no_grad():
        features = model.get_backbone_features(x)
        coords = model(x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Feature shape: {features.shape}")
    print(f"  Output coordinates: {coords.shape}")
    
    assert features.dim() == 4, "Features should be 4D tensor"
    print("  ✅ Feature extraction works")


def test_freeze_backbone():
    """Test freezing backbone parameters."""
    print("\n" + "=" * 60)
    print("Testing Backbone Freezing")
    print("=" * 60)
    
    model_frozen = create_model(
        backbone="efficientnet_b0",
        pretrained=True,
        freeze_backbone=True
    )
    model_unfrozen = create_model(
        backbone="efficientnet_b0",
        pretrained=True,
        freeze_backbone=False
    )
    
    # Count trainable parameters
    frozen_trainable = sum(p.numel() for p in model_frozen.parameters() if p.requires_grad)
    unfrozen_trainable = sum(p.numel() for p in model_unfrozen.parameters() if p.requires_grad)
    
    print(f"  Trainable params (frozen backbone): {frozen_trainable:,}")
    print(f"  Trainable params (unfrozen backbone): {unfrozen_trainable:,}")
    
    assert frozen_trainable < unfrozen_trainable, \
        "Frozen model should have fewer trainable parameters"
    print("  ✅ Backbone freezing works correctly")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("AGENT 1 MODEL ARCHITECTURE TESTS")
    print("=" * 60)
    
    try:
        model = test_model_creation()
        test_forward_pass()
        test_device_placement()
        test_backbone_features()
        test_freeze_backbone()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nModel architecture is working correctly.")
        print("You can now use it in your training script.")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

