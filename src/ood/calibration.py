"""Versioned frozen calibration and locked-checkpoint provenance checks."""

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import torch

from src.ood.ood_detector import OODDetector, OODThresholds

PREPROCESSING = {
    "grayscale": "L", "image_size": [150, 150], "resize": "PIL bilinear",
    "normalize_mean": 0.5, "normalize_std": 0.5, "augmentation": False,
}
FEATURE_LAYER = "spatial_pool(blocks(stem(input))).flatten(1); 128 x 2 x 2 = 512"


def checkpoint_sha256(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def validate_splits(splits: dict) -> None:
    groups = []
    for name in ("train", "validation", "test"):
        indices = splits[name]
        if not indices or any(type(i) is not int or i < 0 for i in indices):
            raise ValueError(f"Invalid {name} split indices.")
        if len(set(indices)) != len(indices):
            raise ValueError(f"Duplicate {name} split indices.")
        groups.append(set(indices))
    if any(groups[i] & groups[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Calibration splits overlap, including final test indices.")


def save_calibration(detector: OODDetector, path: Path, checkpoint: Path,
                     split_path: Path, splits: dict) -> None:
    validate_splits(splits)
    artifact = {
        "schema_version": 1,
        "feature_mean": detector.feature_mean,
        "feature_variance": detector.feature_variance,
        "thresholds": asdict(detector.thresholds),
        "preprocessing": PREPROCESSING,
        "feature_layer": FEATURE_LAYER,
        "checkpoint_path": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": checkpoint_sha256(checkpoint),
        "splits": {
            name: {"reference": str(Path(split_path).resolve()),
                   "reference_sha256": checkpoint_sha256(split_path),
                   "count": len(splits[name]), "indices": splits[name]}
            for name in ("train", "validation")
        },
    }
    _validate_artifact(artifact, checkpoint)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, path)
    metadata = {k: v for k, v in artifact.items() if k not in {"feature_mean", "feature_variance"}}
    metadata["feature_dimension"] = artifact["feature_mean"].numel()
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def _validate_artifact(artifact: dict, checkpoint: Path) -> None:
    if artifact["schema_version"] != 1:
        raise ValueError("Unsupported calibration schema.")
    if artifact["preprocessing"] != PREPROCESSING or artifact["feature_layer"] != FEATURE_LAYER:
        raise ValueError("Calibration preprocessing or feature layer mismatch.")
    if Path(artifact["checkpoint_path"]).resolve() != Path(checkpoint).resolve():
        raise ValueError("Calibration checkpoint path mismatch.")
    if artifact["checkpoint_sha256"] != checkpoint_sha256(checkpoint):
        raise ValueError("Checkpoint hash mismatch.")
    for name in ("feature_mean", "feature_variance"):
        value = artifact[name]
        if not isinstance(value, torch.Tensor) or value.shape != (512,) or not torch.isfinite(value).all():
            raise ValueError(f"Invalid calibration {name}.")
    if not (artifact["feature_variance"] > 0).all():
        raise ValueError("Calibration feature variance must be positive.")
    thresholds = OODThresholds(**artifact["thresholds"])
    if not all(math.isfinite(v) for v in asdict(thresholds).values()):
        raise ValueError("Non-finite calibration thresholds.")
    if not 0.5 < thresholds.validation_percentile < 1 or thresholds.feature_distance_threshold < 0:
        raise ValueError("Invalid calibration thresholds.")
    groups = []
    for name in ("train", "validation"):
        split = artifact["splits"][name]
        indices = split["indices"]
        if (not split["reference"] or len(split["reference_sha256"]) != 64
                or not indices or split["count"] != len(indices)
                or any(type(i) is not int or i < 0 for i in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError(f"Invalid calibration {name} provenance.")
        groups.append(set(indices))
    if groups[0] & groups[1]:
        raise ValueError("Calibration train/validation overlap.")
    split_path = Path(artifact["splits"]["train"]["reference"]).resolve()
    if Path(artifact["splits"]["validation"]["reference"]).resolve() != split_path:
        raise ValueError("Calibration split reference mismatch.")
    # Hash and parse the same bytes; inspect test indices only for disjointness.
    split_bytes = split_path.read_bytes()
    split_digest = hashlib.sha256(split_bytes).hexdigest()
    for name in ("train", "validation"):
        if artifact["splits"][name]["reference_sha256"] != split_digest:
            raise ValueError(f"Calibration {name} split file hash mismatch.")
    current_splits = json.loads(split_bytes)
    validate_splits(current_splits)
    for name in ("train", "validation"):
        if current_splits[name] != artifact["splits"][name]["indices"]:
            raise ValueError(f"Calibration {name} indices mismatch with current split file.")


def load_calibration(path: Path, checkpoint: Path) -> dict:
    """Load only tensors/primitives and reject incompatible or invalid artifacts."""
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    _validate_artifact(artifact, checkpoint)
    return artifact
