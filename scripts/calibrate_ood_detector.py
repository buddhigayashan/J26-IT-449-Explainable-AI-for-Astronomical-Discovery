"""Explicitly freeze train statistics and validation thresholds; never read test samples."""

import argparse
import json
from pathlib import Path
import sys

import torch
import yaml
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.download_mirabest import load_mirabest
from src.data.split_dataset import TransformedSubset
from src.inference.predict import CHECKPOINT_PATH, CALIBRATION_PATH, load_checkpoint
from src.ood.calibration import checkpoint_sha256, save_calibration, validate_splits
from src.ood.ood_detector import OODDetector


def deterministic_transform(image):
    """Preprocess trusted ID samples without external-upload rejection rules."""
    grayscale = image.convert("L").resize((150, 150), Image.Resampling.BILINEAR)
    tensor = transforms.ToTensor()(grayscale)
    return transforms.Normalize((0.5,), (0.5,))(tensor)


def calibrate_from_splits(model, dataset, splits, percentile=0.99, batch_size=32):
    validate_splits(splits)
    detector = OODDetector(model, batch_size=batch_size)
    detector.fit_feature_statistics(TransformedSubset(dataset, splits["train"], deterministic_transform))
    detector.calibrate(TransformedSubset(dataset, splits["validation"], deterministic_transform), percentile)
    return detector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--output", type=Path, default=CALIBRATION_PATH)
    parser.add_argument("--percentile", type=float, default=0.99)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    split_path = PROJECT_ROOT / config["paths"]["splits"] / "split_indices.json"
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    validate_splits(splits)
    torch.manual_seed(int(config["training"]["seed"]))
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    digest = checkpoint_sha256(CHECKPOINT_PATH)
    load_checkpoint.cache_clear()
    model = load_checkpoint(str(CHECKPOINT_PATH))
    dataset = load_mirabest(PROJECT_ROOT / config["paths"]["mirabest_root"], download=False)
    detector = calibrate_from_splits(model, dataset, splits, args.percentile,
                                    int(config["training"]["batch_size"]))
    if checkpoint_sha256(CHECKPOINT_PATH) != digest:
        raise ValueError("Checkpoint changed during calibration.")
    save_calibration(detector, args.output, CHECKPOINT_PATH, split_path, splits)
    print(f"Frozen OOD calibration saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
