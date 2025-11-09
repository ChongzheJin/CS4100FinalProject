import json
import math
from pathlib import Path

import yaml
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

from models.agent1_model import create_model
from src.utils.coordinates import compute_normalization_params
from dataio.datasets import GeoCSVDataset

ROOT = Path(__file__).resolve().parents[1]

def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

# Read configs
data_cfg = load_yaml(ROOT / "configs" / "data.yaml")
eval_cfg = load_yaml(ROOT / "configs" / "eval_agent1.yaml")

img_size = int(data_cfg.get("img_size", 256))
batch_size = int(eval_cfg.get("batch_size_override") or data_cfg.get("batch_size", 64))
num_workers = int(eval_cfg.get("num_workers_override") or data_cfg.get("num_workers", 4))
image_root : Path = data_cfg.get("image_root")
test_csv : Path = data_cfg.get("test_csv")
train_csv : Path = data_cfg.get("train_csv")
checkpoint_path = eval_cfg.get("checkpoint") or "checkpoints/agent1/best_checkpoint.pt"
output_dir = ROOT / (eval_cfg.get("output_dir") or "outputs/eval_agent1")
output_dir.mkdir(parents=True, exist_ok=True)
limit = eval_cfg.get("limit")  # None or int

# Device & DataLoader helpers
def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

device = pick_device()

def make_loader_kwargs(bz: int, nw: int, d: torch.device) -> dict:
    on_gpu = (d.type == "cuda")
    on_mps = (d.type == "mps")
    pin_memory = bool(on_gpu)  # CUDA True
    persistent_workers = (nw > 0) and (on_gpu or on_mps)
    return dict(
        batch_size=bz,
        num_workers=nw,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

loader_kwargs = make_loader_kwargs(batch_size, num_workers, device)

# Helpers
def _seed_all(seed: int = 2025):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _denorm_from_minus1_1(x: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    return (x + 1.0) * 0.5 * (vmax - vmin) + vmin

def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0088
    p = math.pi / 180.0
    lat1, lon1, lat2, lon2 = lat1 * p, lon1 * p, lat2 * p, lon2 * p
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    c = 2 * math.asin(math.sqrt(a))
    return R * c

class _EvalDataset(GeoCSVDataset):
    def __init__(self, csv_path: Path, image_size: int, l: int | None = None):
        super().__init__(
            csv_path=str(csv_path),
            img_size=image_size,
            train=False,
            normalize=False,
        )
        if l is not None:
            self.df = self.df.head(int(l)).reset_index(drop=True)

        # Save image paths(name), for later evaltion output
        self._image_paths = self.df["image_path"].astype(str).tolist()

    def __getitem__(self, idx: int):
        coords : torch.Tensor # coords: [lat, lon]
        image_tensor, coords = super().__getitem__(idx)
        lat = float(coords[0])
        lon = float(coords[1])
        filename = os.path.basename(self._image_paths[idx])
        return image_tensor, lat, lon, filename

# Main eval method
def run_eval():
    _seed_all(2025)
    _ensure_dir(output_dir)

    # Paths checking
    if not train_csv:
        raise ValueError("configs/data.yaml missing train_csv")
    if not test_csv:
        raise ValueError("configs/data.yaml missing test_csv")
    train_csv_path = ROOT / train_csv
    test_csv_path  = ROOT / test_csv
    if not train_csv_path.is_file():
        raise FileNotFoundError(f"train_csv not found: {train_csv_path}")
    if not test_csv_path.is_file():
        raise FileNotFoundError(f"test_csv not found: {test_csv_path}")

    # Normalized parameters (consistent with training)
    lat_min, lat_max, lon_min, lon_max = compute_normalization_params(str(train_csv_path))

    # DataLoader
    ds = _EvalDataset(test_csv_path, img_size, l=limit)
    loader = torch.utils.data.DataLoader(ds, shuffle=False, **loader_kwargs)

    # Device information
    print(f"[Info] Device: {device.type}")

    # Read train config
    train_cfg_path = ROOT / "configs" / "train_agent1.yaml"
    if not train_cfg_path.is_file():
        raise FileNotFoundError(f"Train config not found: {train_cfg_path}")
    train_cfg = load_yaml(train_cfg_path)
    model_cfg = train_cfg["model"]

    # Create model
    model = create_model(
        backbone=model_cfg["backbone"],
        pretrained=model_cfg["pretrained"],
        dropout=model_cfg["dropout"],
        freeze_backbone=model_cfg["freeze_backbone"],
        device=device.type,
    )
    model.to(device=device, dtype=torch.float32)

    ckpt_path = ROOT / checkpoint_path
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    pred_lat_all, pred_lon_all = [], []
    gt_lat_all, gt_lon_all = [], []
    fname_all, km_all = [], []

    with torch.no_grad():
        for xb, lat_gt, lon_gt, fnames in tqdm(loader, desc="Evaluating", ncols=90):
            xb = xb.to(device, non_blocking=loader_kwargs["pin_memory"])
            y = model(xb)
            y = y.detach().cpu().numpy()
            pred_lat = _denorm_from_minus1_1(y[:, 0], lat_min, lat_max)
            pred_lon = _denorm_from_minus1_1(y[:, 1], lon_min, lon_max)

            pred_lat_all.extend(pred_lat.tolist())
            pred_lon_all.extend(pred_lon.tolist())
            gt_lat_all.extend(lat_gt.numpy().tolist())
            gt_lon_all.extend(lon_gt.numpy().tolist())
            fname_all.extend(list(fnames))

    # Error
    abs_lat_err = np.abs(np.array(pred_lat_all) - np.array(gt_lat_all))
    abs_lon_err = np.abs(np.array(pred_lon_all) - np.array(gt_lon_all))
    for plat, plon, glat, glon in zip(pred_lat_all, pred_lon_all, gt_lat_all, gt_lon_all):
        km_all.append(_haversine_km(plat, plon, glat, glon))
    km_all = np.array(km_all, dtype=np.float64)

    # Output Table
    table = pd.DataFrame({
        "filename": fname_all,
        "pred_lat": pred_lat_all,
        "pred_lon": pred_lon_all,
        "gt_lat": gt_lat_all,
        "gt_lon": gt_lon_all,
        "abs_lat_err": abs_lat_err,
        "abs_lon_err": abs_lon_err,
        "km_error": km_all,
    })

    csv_path = output_dir / "predictions.csv"
    table.to_csv(csv_path, index=False)

    metrics = {
        "count": int(len(table)),
        "mean_abs_lat_err": float(abs_lat_err.mean()),
        "mean_abs_lon_err": float(abs_lon_err.mean()),
        "mean_km_error": float(km_all.mean()),
        "median_km_error": float(np.median(km_all)),
        "ckpt": str(ckpt_path.relative_to(ROOT)) if str(ckpt_path).startswith(str(ROOT)) else str(ckpt_path),
        "test_csv": str(test_csv_path.relative_to(ROOT)),
        "train_csv": str(train_csv_path.relative_to(ROOT)),
        "img_size": img_size,
        "batch_size": batch_size,
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Summary Output
    print("\n=== Eval Summary ===")
    print(f"Samples: {metrics['count']}")
    print(f"Mean Latitude error: {metrics['mean_abs_lat_err']:.6f} degrees")
    print(f"Mean Longitude error: {metrics['mean_abs_lon_err']:.6f} degrees")
    print(f"Mean geodesic error: {metrics['mean_km_error']:.3f} km")
    print(f"Median geodesic error: {metrics['median_km_error']:.3f} km")
    print(f"Saved table: {csv_path}")
    print(f"Saved metrics: {output_dir / 'metrics.json'}")

if __name__ == "__main__":
    run_eval()
