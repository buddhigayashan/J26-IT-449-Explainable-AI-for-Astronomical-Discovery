"""Calibrate only the isolated AUTO synthetic-positive gate using TRAIN/VALIDATION."""

import argparse
import json
from pathlib import Path
import sys

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.download_mirabest import load_mirabest
from src.inference.predict import CHECKPOINT_PATH, load_checkpoint
from src.ood.calibration import checkpoint_sha256, validate_splits
from src.ood.external_compatibility import (
    EXTERNAL_CALIBRATION_PATH, calibrate_external_from_splits, save_external_calibration,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    split_path = PROJECT_ROOT / config["paths"]["splits"] / "split_indices.json"
    splits = json.loads(split_path.read_bytes())
    validate_splits(splits)
    seed = int(config["training"]["seed"])
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    protected = [CHECKPOINT_PATH, PROJECT_ROOT / "outputs/ood/ood_calibration.pt",
                 PROJECT_ROOT / "outputs/ood/ood_calibration.json", split_path,
                 PROJECT_ROOT / "outputs/metrics/resnet_test_evaluation.json",
                 PROJECT_ROOT / "outputs/metrics/final_test_metrics.csv"]
    before = {path: checkpoint_sha256(path) for path in protected}
    dataset_root = PROJECT_ROOT / config["paths"]["mirabest_root"]
    dataset = load_mirabest(dataset_root, download=False)
    model = load_checkpoint(str(CHECKPOINT_PATH))
    print(f"Preparing fixed positive variants: TRAIN={len(splits['train'])}, VALIDATION={len(splits['validation'])}; no TEST sample access.", flush=True)
    detector, counts = calibrate_external_from_splits(model, dataset, splits, batch_size=int(config["training"]["batch_size"]))
    if any(checkpoint_sha256(path) != digest for path, digest in before.items()):
        raise ValueError("Protected artifact changed during external calibration.")
    manifest = dataset_root / "batches/hash.md5"
    provenance = {"loader": "verified load_mirabest(download=False)", "root": str(dataset_root.resolve()),
                  "manifest_sha256": checkpoint_sha256(manifest)}
    save_external_calibration(detector, EXTERNAL_CALIBRATION_PATH, CHECKPOINT_PATH,
                              split_path, splits, counts, seed, provenance)
    print(f"External calibration: {EXTERNAL_CALIBRATION_PATH}")
    print(f"External thresholds: {detector.thresholds}")
    print(json.dumps(counts, indent=2))
    print("Protected checkpoint, original OOD artifacts, split file, and final metrics: SHA256 unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
