from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

ROOT = Path(__file__).resolve().parents[2]

def default_transforms(img_size = 256, train = True) -> A.Compose:
    """
    Build albumentations transforms for training or eval

    :param img_size: Target square size after pad
    :param train: Determine whether to use train branch
    :return: Albumentations transforms
    """
    if train:
        aug = A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0),
            # Optimize the agent's handling of different weather conditions
            A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
            # Note: HorizontalFlip removed - it reverses text/signs and directional cues
            # Normalize to [0, 1] and convert to float32 for MPS compatibility
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
            ToTensorV2()
        ])
    else:
        aug = A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0),
            # Normalize to [0, 1] and convert to float32 for MPS compatibility
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
            ToTensorV2()
        ])
    return aug


def _to_abs(path_str: str) -> str:
    # turn relative CSV path into absolute path under project ROOT
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    else:
        return str((ROOT / p).resolve())


class GeoCSVDataset(Dataset):
    """
    Dataset that loads images and information of location(lat, lon) from a csv file.
    
    Coordinates are normalized to [-1, 1] range for training stability.
    CSV format:
        - image_path: path relative to project root (recommended, should work even not)
        - lat, lon : latitude and longitude in degrees
    """
    def __init__(
        self,
        csv_path: str,
        img_size: int = 256,
        train: bool = True,
        lat_min: float = None,
        lat_max: float = None,
        lon_min: float = None,
        lon_max: float = None,
        normalize: bool = True
    ):
        """
        Initialize the dataset with a CSV file path, image size, and training mode.

        :param csv_path: Path to the CSV file containing image paths and location information.
        :param img_size: Target square size after padding.
        :param train: Determine whether to use train branch.
        :param lat_min: Minimum latitude for normalization (required if normalize=True)
        :param lat_max: Maximum latitude for normalization (required if normalize=True)
        :param lon_min: Minimum longitude for normalization (required if normalize=True)
        :param lon_max: Maximum longitude for normalization (required if normalize=True)
        :param normalize: Whether to normalize coordinates to [-1, 1] range
        """
        self.df = pd.read_csv(_to_abs(csv_path))
        self.transforms = default_transforms(img_size, train)
        self.normalize = normalize
        
        if normalize:
            if any(x is None for x in [lat_min, lat_max, lon_min, lon_max]):
                raise ValueError(
                    "Normalization parameters (lat_min, lat_max, lon_min, lon_max) "
                    "are required when normalize=True. "
                    "Use compute_normalization_params() from utils.coordinates to compute them."
                )
            self.lat_min = lat_min
            self.lat_max = lat_max
            self.lon_min = lon_min
            self.lon_max = lon_max

    def __len__(self):
        """
        :return: Number of samples.
        """
        return len(self.df)

    def __getitem__(self, item: int):
        """
        :param item: Index of the sample to retrieve.
        :return: Tuple containing the processed image and corresponding normalized coordinates.
                 Coordinates are in [-1, 1] range if normalize=True, otherwise in degrees.
        """
        row = self.df.iloc[item]
        # Open the image and convert to RGB
        img = Image.open(_to_abs(str(row["image_path"]))).convert("RGB")
        # Convert to an array
        img = np.array(img)
        # Apply the transforms
        img = self.transforms(image=img)["image"]

        # Get coordinates in degrees (CSV already has degrees)
        lat_deg = float(row["lat"])
        lon_deg = float(row["lon"])
        
        # Normalize coordinates to [-1, 1] range if requested
        if self.normalize:
            import sys
            from pathlib import Path
            # Add src to path if not already there
            src_path = Path(__file__).resolve().parent.parent
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))
            from src.utils.coordinates import normalize_coordinates
            
            coords_deg = torch.tensor([[lat_deg, lon_deg]], dtype=torch.float32)
            coords_norm = normalize_coordinates(
                coords_deg,
                self.lat_min,
                self.lat_max,
                self.lon_min,
                self.lon_max
            )
            return img, coords_norm.squeeze(0)  # Remove batch dimension
        else:
            # Return raw degrees (for compatibility or debugging)
            return img, torch.tensor([lat_deg, lon_deg], dtype=torch.float32)