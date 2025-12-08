"""
Coordinate normalization and denormalization utilities.

Handles conversion between:
- Degrees (raw lat/lon from CSV)
- Radians (for some calculations)
- Normalized [-1, 1] range (for training stability)
"""

import torch
import numpy as np
from typing import Tuple


def normalize_coordinates(
    coords: torch.Tensor,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float
) -> torch.Tensor:
    """
    Normalize coordinates from degrees to [-1, 1] range.
    
    Args:
        coords: Tensor of shape [batch_size, 2] where columns are [lat, lon] in degrees
        lat_min: Minimum latitude in training data
        lat_max: Maximum latitude in training data
        lon_min: Minimum longitude in training data
        lon_max: Maximum longitude in training data
    
    Returns:
        Normalized coordinates in [-1, 1] range, shape [batch_size, 2]
    """
    lat = coords[:, 0]
    lon = coords[:, 1]
    
    # Normalize to [0, 1] then to [-1, 1]
    lat_norm = 2 * (lat - lat_min) / (lat_max - lat_min) - 1
    lon_norm = 2 * (lon - lon_min) / (lon_max - lon_min) - 1
    
    return torch.stack([lat_norm, lon_norm], dim=1)


def denormalize_coordinates(
    normalized_coords: torch.Tensor,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float
) -> torch.Tensor:
    """
    Denormalize coordinates from [-1, 1] range back to degrees.
    
    Args:
        normalized_coords: Tensor of shape [batch_size, 2] in [-1, 1] range
        lat_min: Minimum latitude used for normalization
        lat_max: Maximum latitude used for normalization
        lon_min: Minimum longitude used for normalization
        lon_max: Maximum longitude used for normalization
    
    Returns:
        Coordinates in degrees, shape [batch_size, 2]
    """
    lat_norm = normalized_coords[:, 0]
    lon_norm = normalized_coords[:, 1]
    
    # Denormalize from [-1, 1] to [0, 1] then to original range
    lat = (lat_norm + 1) / 2 * (lat_max - lat_min) + lat_min
    lon = (lon_norm + 1) / 2 * (lon_max - lon_min) + lon_min
    
    return torch.stack([lat, lon], dim=1)


def compute_normalization_params(csv_path: str) -> Tuple[float, float, float, float]:
    """
    Compute min/max lat/lon from training CSV for normalization.
    
    Args:
        csv_path: Path to CSV file with lat/lon columns
    
    Returns:
        Tuple of (lat_min, lat_max, lon_min, lon_max)
    """
    import pandas as pd
    from pathlib import Path
    
    df = pd.read_csv(csv_path)
    
    lat_min = float(df["lat"].min())
    lat_max = float(df["lat"].max())
    lon_min = float(df["lon"].min())
    lon_max = float(df["lon"].max())
    
    return lat_min, lat_max, lon_min, lon_max

