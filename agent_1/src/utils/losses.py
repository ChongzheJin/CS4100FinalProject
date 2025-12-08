"""
Loss functions for Agent 1 - Direct coordinate regression.

Primary loss: Haversine distance (great-circle distance on Earth's surface).
"""

import torch
import torch.nn as nn
from math import radians, sin, cos, asin, sqrt


def haversine_distance_km(
    lat1: torch.Tensor,
    lon1: torch.Tensor,
    lat2: torch.Tensor,
    lon2: torch.Tensor
) -> torch.Tensor:
    """
    Calculate Haversine distance between two sets of coordinates.
    
    Args:
        lat1: Latitude of first point(s) in degrees
        lon1: Longitude of first point(s) in degrees
        lat2: Latitude of second point(s) in degrees
        lon2: Longitude of second point(s) in degrees
    
    Returns:
        Distance in kilometers, same shape as input tensors
    """
    # Convert degrees to radians
    lat1_rad = torch.deg2rad(lat1)
    lon1_rad = torch.deg2rad(lon1)
    lat2_rad = torch.deg2rad(lat2)
    lon2_rad = torch.deg2rad(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1_rad) * torch.cos(lat2_rad) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.asin(torch.sqrt(a))
    
    # Earth radius in kilometers
    R = 6371.0
    distance_km = R * c
    
    return distance_km


class HaversineLoss(nn.Module):
    """
    Haversine distance loss for coordinate regression.
    
    Computes the great-circle distance between predicted and true coordinates
    on Earth's surface. Handles normalization/denormalization automatically.
    """
    
    def __init__(
        self,
        lat_min: float = None,
        lat_max: float = None,
        lon_min: float = None,
        lon_max: float = None,
        normalize: bool = True
    ):
        """
        Initialize Haversine loss.
        
        Args:
            lat_min: Minimum latitude for denormalization (required if normalize=True)
            lat_max: Maximum latitude for denormalization (required if normalize=True)
            lon_min: Minimum longitude for denormalization (required if normalize=True)
            lon_max: Maximum longitude for denormalization (required if normalize=True)
            normalize: Whether inputs are normalized to [-1, 1] range
        """
        super().__init__()
        self.normalize = normalize
        
        if normalize:
            if any(x is None for x in [lat_min, lat_max, lon_min, lon_max]):
                raise ValueError("Normalization parameters required when normalize=True")
            self.register_buffer('lat_min', torch.tensor(lat_min))
            self.register_buffer('lat_max', torch.tensor(lat_max))
            self.register_buffer('lon_min', torch.tensor(lon_min))
            self.register_buffer('lon_max', torch.tensor(lon_max))
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Haversine loss.
        
        Args:
            predictions: Predicted coordinates, shape [batch_size, 2] where columns are [lat, lon]
            targets: True coordinates, shape [batch_size, 2] where columns are [lat, lon]
        
        Returns:
            Mean Haversine distance in kilometers (scalar tensor)
        """
        # Denormalize if needed
        if self.normalize:
            from .coordinates import denormalize_coordinates
            
            pred_deg = denormalize_coordinates(
                predictions,
                self.lat_min.item(),
                self.lat_max.item(),
                self.lon_min.item(),
                self.lon_max.item()
            )
            target_deg = denormalize_coordinates(
                targets,
                self.lat_min.item(),
                self.lat_max.item(),
                self.lon_min.item(),
                self.lon_max.item()
            )
        else:
            pred_deg = predictions
            target_deg = targets
        
        # Extract lat/lon
        pred_lat = pred_deg[:, 0]
        pred_lon = pred_deg[:, 1]
        target_lat = target_deg[:, 0]
        target_lon = target_deg[:, 1]
        
        # Compute Haversine distance
        distances = haversine_distance_km(pred_lat, pred_lon, target_lat, target_lon)
        
        # Return mean distance
        return distances.mean()

