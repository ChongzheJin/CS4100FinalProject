from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from math import radians

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
            # If the agent's accuracy is not enough, prioritize removing it.
            A.HorizontalFlip(p=0.5),
            # Ready for PyTorch
            ToTensorV2()
        ])
    else:
        aug = A.Compose([
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0),
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

    CSV format:
        - image_path: path relative to project root (recommended, should work even not)
        - lat, lon : positive and negative float
    """
    def __init__(self, csv_path : str, img_size : int = 256, train : bool = True):
        """
        Initialize the dataset with a CSV file path, image size, and training mode.

        :param csv_path: Path to the CSV file containing image paths and location information.
        :param img_size: Target square size after padding.
        :param train: Determine whether to use train branch.
        """
        self.df = pd.read_csv(_to_abs(csv_path))
        self.transforms = default_transforms(img_size, train)

    def __len__(self):
        """
        :return: Number of samples.
        """
        return len(self.df)

    def __getitem__(self, item : int):
        """
        :param item: Index of the sample to retrieve.
        :return: Tuple containing the processed image and corresponding geographical coordinates.
        """
        row = self.df.iloc[item]
        # Open the image and convert to RGB
        img = Image.open(_to_abs(str(row["image_path"]))).convert("RGB")
        # Convert to an array
        img = np.array(img)
        # Apply the transforms
        img = self.transforms(image=img)["image"]

        # TODO: Due to the differing scales of lat and lon in projection, an attempt is made to convert
        #  them into unit sphere vectors (x, y, z).
        lat_rad = torch.tensor(radians(float(row["lat"])), dtype=torch.float32)
        lon_rad = torch.tensor(radians(float(row["lon"])), dtype=torch.float32)
        return img, torch.stack([lat_rad, lon_rad], dim=0)