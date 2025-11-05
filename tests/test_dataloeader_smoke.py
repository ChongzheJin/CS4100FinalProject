import math
from pathlib import Path
from torch.utils.data import DataLoader
from src.dataio.datasets import GeoCSVDataset

ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = ROOT / "data" / "processed" / "us_streetview" / "train.csv"

def test_smoke_dataloader():
    ds = GeoCSVDataset(str(TRAIN_CSV), img_size=256, train=True)
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)

    x, y = next(iter(dl))

    # Assert input images have 4 dimensions and 3 channels
    assert x.ndim == 4 and x.shape[1] == 3
    # Assert labels have 2 elements per sample (latitude and longitude)
    assert y.shape[-1] == 2

    # TODO: change it into unit sphere vector (x, y, z)
    # Assert latitude values are within the valid range with small epsilon tolerance
    assert float(y[:,0].min()) >= -math.pi/2 - 1e-6
    assert float(y[:,0].max()) <=  math.pi/2 + 1e-6