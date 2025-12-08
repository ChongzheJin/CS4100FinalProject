
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

from grid_mapper import latlon_to_grid



#Transforms the dataset from (IMG, lat, long) into the tuple form (IMG, grid_label)
class StreetViewGridDataset(Dataset):
    """
    Dataset for Agent 2 — loads images and maps (lat, lon) → grid label.
    """
    def __init__(self, csv_path, img_size=256, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform or transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

        # compute grid labels from coordinates
        self.df["grid_label"] = self.df.apply(
            lambda r: latlon_to_grid(r["lat"], r["lon"]), axis=1
        )
        self.df = self.df[self.df["grid_label"] >= 0].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        img = self.transform(img)
        label = int(row["grid_label"])
        return img, label
