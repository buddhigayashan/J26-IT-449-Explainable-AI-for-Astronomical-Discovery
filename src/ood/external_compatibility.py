"""Separate synthetic-domain feature gate for AUTO; no classifier training."""

from dataclasses import asdict
import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.ood.auto_radio import DOMAIN_RULES, PREPROCESSING_CONFIGURATION, PREPROCESSING_VERSION, extract_auto_candidates
from src.ood.calibration import FEATURE_LAYER, checkpoint_sha256, validate_splits
from src.ood.ood_detector import OODDetector, OODThresholds

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_CALIBRATION_PATH = PROJECT_ROOT / "outputs/ood/external_compatibility_calibration.pt"
TRANSFORMATIONS = {
    "original": "original 8-bit L pixels",
    "wide_canvas": "180x150 centered canvas; median-border background; full frame; bilinear to 150",
    "tall_canvas": "150x180 centered canvas; median-border background; full frame; bilinear to 150",
    "resolution_96": "PIL bilinear 150->96->150; no cropping",
    "contrast_0.9": "multiply intensity by 0.9; round to uint8",
    "contrast_1.1": "multiply intensity by 1.1; clip [0,255]; round to uint8",
    "gamma_0.9": "255*(pixel/255)**0.9; round to uint8",
    "gamma_1.1": "255*(pixel/255)**1.1; round to uint8",
    "rgb_encoding": "RGB=(pixel, round(0.75*pixel), round(0.5*pixel)); affine shared morphology",
    "jpeg_95": "PIL grayscale JPEG quality=95; decode; no cropping",
}
CONSISTENCY_POLICY = {"min_external_confidence": 0.60, "agreement": "unanimous",
                      "gate_policy": "all plausible candidates must pass", "aggregation": "mean probabilities"}


def external_style_variants(image: Image.Image):
    """Fixed, mild positive transforms; no random selection or FR-label filtering."""
    gray = image.convert("L")
    if gray.size != (150, 150):
        raise ValueError("Calibration requires original 150x150 MiraBest images.")
    pixels = np.asarray(gray, dtype=np.float64)
    border = np.concatenate((pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]))
    background = int(np.rint(np.median(border)))
    yield "original", gray
    for name, size in (("wide_canvas", (180, 150)), ("tall_canvas", (150, 180))):
        canvas = Image.new("L", size, background)
        canvas.paste(gray, ((size[0] - 150) // 2, (size[1] - 150) // 2))
        yield name, canvas
    yield "resolution_96", gray.resize((96, 96), Image.Resampling.BILINEAR).resize((150, 150), Image.Resampling.BILINEAR)
    for factor in (0.9, 1.1):
        yield f"contrast_{factor}", Image.fromarray(np.rint(np.clip(pixels * factor, 0, 255)).astype(np.uint8))
    for gamma in (0.9, 1.1):
        yield f"gamma_{gamma}", Image.fromarray(np.rint(255 * (pixels / 255) ** gamma).astype(np.uint8))
    rgb = np.stack((pixels, np.rint(pixels * 0.75), np.rint(pixels * 0.5)), axis=-1).astype(np.uint8)
    yield "rgb_encoding", Image.fromarray(rgb)
    buffer = io.BytesIO()
    gray.save(buffer, format="JPEG", quality=95)
    buffer.seek(0)
    with Image.open(buffer) as encoded:
        yield "jpeg_95", encoded.copy()


class PositiveVariantDataset(Dataset):
    """Precompute accepted candidates from explicitly allowed split indices only."""
    def __init__(self, dataset, indices):
        self.tensors = []
        self.counts = {name: {"images_accepted": 0, "images_rejected": 0, "candidate_count": 0}
                       for name in TRANSFORMATIONS}
        for index in indices:
            image, _ = dataset[index]  # Never access or use labels to select positives.
            for name, variant in external_style_variants(image):
                try:
                    prepared = extract_auto_candidates(variant)
                except ValueError:
                    self.counts[name]["images_rejected"] += 1
                    continue
                self.counts[name]["images_accepted"] += 1
                self.counts[name]["candidate_count"] += len(prepared.candidates)
                self.tensors.extend(c.guarded.tensor[0] for c in prepared.candidates)
        if not self.tensors:
            raise ValueError("No compatible positive variants in the supplied split.")

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, index):
        return self.tensors[index], 0


def calibrate_external_from_splits(model, dataset, splits, percentile=0.99, batch_size=32):
    """TRAIN variant feature reference; VALIDATION variant score quantiles only."""
    validate_splits(splits)  # Inspect TEST indices only to enforce disjointness.
    if not 0.5 < percentile < 1:
        raise ValueError("Validation percentile must be between 0.5 and 1.")
    train = PositiveVariantDataset(dataset, splits["train"])
    validation = PositiveVariantDataset(dataset, splits["validation"])
    detector = OODDetector(model, batch_size=batch_size)
    detector.fit_feature_statistics(train)
    detector.calibrate(validation, percentile)
    return detector, {"train": train.counts, "validation": validation.counts}


def save_external_calibration(detector, path, checkpoint, split_path, splits, counts, seed=42, dataset_provenance=None):
    validate_splits(splits)
    path = Path(path)
    # Only explicitly named external artifacts may be written, never original OOD files.
    if path.name != "external_compatibility_calibration.pt":
        raise ValueError("External calibration must use its isolated external_compatibility_calibration.pt filename.")
    artifact = {
        "schema_version": 1, "kind": "external_synthetic_compatibility",
        "feature_mean": detector.feature_mean, "feature_variance": detector.feature_variance,
        "feature_dimension": 512, "feature_layer": FEATURE_LAYER,
        "thresholds": asdict(detector.thresholds), "preprocessing_version": PREPROCESSING_VERSION,
        "preprocessing": PREPROCESSING_CONFIGURATION,
        "domain_rules": DOMAIN_RULES, "transformations": TRANSFORMATIONS,
        "consistency_policy": CONSISTENCY_POLICY, "seed": seed,
        "checkpoint_path": str(Path(checkpoint).resolve()), "checkpoint_sha256": checkpoint_sha256(checkpoint),
        "split_reference": str(Path(split_path).resolve()), "split_reference_sha256": checkpoint_sha256(split_path),
        "splits": {name: {"count": len(splits[name]), "indices": list(splits[name])}
                   for name in ("train", "validation")},
        "variant_counts": counts, "dataset_provenance": dataset_provenance or {},
        "limitations": "Synthetic positive-only calibration; no measured natural-image false-accept rate or external-survey accuracy.",
    }
    validate_external_artifact(artifact, checkpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, path)
    metadata = {k: v for k, v in artifact.items() if k not in {"feature_mean", "feature_variance"}}
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def validate_external_artifact(artifact, checkpoint):
    """Fail closed on stale policy, feature statistics, model, or split provenance."""
    if artifact["schema_version"] != 1 or artifact["kind"] != "external_synthetic_compatibility":
        raise ValueError("Invalid external compatibility schema.")
    expected = {"feature_dimension": 512, "feature_layer": FEATURE_LAYER,
                "preprocessing_version": PREPROCESSING_VERSION, "domain_rules": DOMAIN_RULES,
                "preprocessing": PREPROCESSING_CONFIGURATION,
                "transformations": TRANSFORMATIONS, "consistency_policy": CONSISTENCY_POLICY}
    if any(artifact[k] != v for k, v in expected.items()):
        raise ValueError("External calibration preprocessing/policy mismatch.")
    if Path(artifact["checkpoint_path"]).resolve() != Path(checkpoint).resolve() or artifact["checkpoint_sha256"] != checkpoint_sha256(checkpoint):
        raise ValueError("External calibration checkpoint mismatch.")
    for name in ("feature_mean", "feature_variance"):
        value = artifact[name]
        if not isinstance(value, torch.Tensor) or value.shape != (512,) or not torch.isfinite(value).all():
            raise ValueError("Invalid external feature statistics.")
    if not (artifact["feature_variance"] > 0).all():
        raise ValueError("Invalid external feature variance.")
    thresholds = OODThresholds(**artifact["thresholds"])
    if not all(math.isfinite(v) for v in asdict(thresholds).values()) or not 0.5 < thresholds.validation_percentile < 1 or thresholds.feature_distance_threshold < 0:
        raise ValueError("Invalid external compatibility thresholds.")
    if set(artifact["splits"]) != {"train", "validation"}:
        raise ValueError("External calibration may contain TRAIN/VALIDATION only.")
    split_path = Path(artifact["split_reference"])
    split_bytes = split_path.read_bytes()
    import hashlib
    if hashlib.sha256(split_bytes).hexdigest() != artifact["split_reference_sha256"]:
        raise ValueError("External split provenance hash mismatch.")
    splits = json.loads(split_bytes)
    validate_splits(splits)
    for name in ("train", "validation"):
        group = artifact["splits"][name]
        if group["indices"] != splits[name] or group["count"] != len(splits[name]):
            raise ValueError("External split indices mismatch.")


def load_external_detector(model, checkpoint, path=EXTERNAL_CALIBRATION_PATH):
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    validate_external_artifact(artifact, checkpoint)
    detector = OODDetector(model)
    detector.feature_mean = artifact["feature_mean"]
    detector.feature_variance = artifact["feature_variance"]
    detector.thresholds = OODThresholds(**artifact["thresholds"])
    return detector
