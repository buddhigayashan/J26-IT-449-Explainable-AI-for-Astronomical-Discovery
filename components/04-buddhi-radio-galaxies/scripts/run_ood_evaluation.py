"""Evaluate the calibrated input-domain/OOD safety layer without retraining."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.metrics import roc_auc_score, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.data.download_mirabest import load_mirabest
from src.inference.predict import CHECKPOINT_PATH, CALIBRATION_PATH, load_calibrated_ood_detector
from src.ood.input_guard import prepare_guarded_image
from scripts.calibrate_ood_detector import deterministic_transform


CSV_FIELDS = (
	"source",
	"sample_id",
	"accepted",
	"ood_status",
	"energy_score",
	"feature_distance",
	"reason",
)


def _paths(output_root: Path) -> dict[str, Path]:
	output_root = Path(output_root)
	return {
		"output_dir": output_root / "ood",
		"json": output_root / "ood" / "ood_evaluation.json",
		"csv": output_root / "ood" / "ood_predictions.csv",
		"figure": output_root / "figures" / "ood_score_distribution.png",
	}


def score_metrics(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
	"""OOD is positive; higher raw scores are more OOD-like for both scores.

	Guard rejects have no numeric score and are excluded from score metrics.
	They remain included in rejection counts.
	"""
	scored = [r for r in records if r.get(field) is not None and np.isfinite(r[field])]
	labels = [int(r["source"] == "ood") for r in scored]
	scores = [r[field] for r in scored]
	metrics = {"auroc": None, "fpr_at_95_tpr": None,
		"scored_count": len(scored), "excluded_count": len(records) - len(scored),
		"score_direction": "higher_is_more_ood"}
	if len(set(labels)) == 2:
		metrics["auroc"] = float(roc_auc_score(labels, scores))
		fpr, tpr, _ = roc_curve(labels, scores)
		metrics["fpr_at_95_tpr"] = float(fpr[np.where(tpr >= 0.95)[0][0]])
	return metrics


def scan_ood_images(ood_test_dir: Path, detector) -> list[dict[str, Any]]:
	"""Record each supported external file exactly once, including guard rejects."""
	ood_records = []
	for path in sorted(ood_test_dir.glob("*")) if ood_test_dir.is_dir() else []:
		if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
			continue
		try:
			guarded = prepare_guarded_image(path)
		except ValueError as error:
			result = {"accepted": False, "ood_status": "out_of_distribution",
				"reason": str(error), "energy_score": None, "feature_distance": None}
		else:
			result = detector.validate_tensor(guarded.tensor)
		ood_records.append(_record("ood", path.name, result))
	return ood_records


def _record(source: str, sample_id: str, result: dict[str, Any]) -> dict[str, Any]:
	return {
		"source": source,
		"sample_id": sample_id,
		"accepted": result["accepted"],
		"ood_status": result["ood_status"],
		"energy_score": result.get("energy_score"),
		"feature_distance": result.get("feature_distance"),
		"reason": result["reason"],
	}


def _save_records(path: Path, records: list[dict[str, Any]]) -> None:
	with path.open("w", newline="", encoding="utf-8") as file_handle:
		writer = csv.DictWriter(file_handle, fieldnames=CSV_FIELDS)
		writer.writeheader()
		writer.writerows(records)


def _plot_scores(records: list[dict[str, Any]], path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	figure, axes = plt.subplots(1, 2, figsize=(11, 4))
	for axis, field in zip(axes, ("energy_score", "feature_distance")):
		for source, color in (("in_distribution", "tab:blue"), ("ood", "tab:red")):
			values = [record[field] for record in records if record["source"] == source and record[field] is not None]
			if values:
				axis.hist(values, bins=15, alpha=0.6, label=source, color=color)
		axis.set_title(field.replace("_", " ").title())
		axis.legend()
	figure.tight_layout()
	figure.savefig(path, dpi=150)
	plt.close(figure)


def run_ood_evaluation(
	config_path: Path,
	output_root: Path,
	ood_test_dir: Path,
	in_distribution_split: str = "test",
) -> dict[str, Any]:
	config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
	split_path = PROJECT_ROOT / config["paths"]["splits"] / "split_indices.json"
	split_indices = json.loads(split_path.read_text(encoding="utf-8"))
	dataset = load_mirabest(PROJECT_ROOT / config["paths"]["mirabest_root"], download=False)
	detector = load_calibrated_ood_detector()
	thresholds = detector.thresholds
	threshold_dict = {
		"energy_threshold": thresholds.energy_threshold,
		"feature_distance_threshold": thresholds.feature_distance_threshold,
		"validation_percentile": thresholds.validation_percentile,
	}

	if in_distribution_split not in {"validation", "test"}:
		raise ValueError("in_distribution_split must be validation or test")
	id_records: list[dict[str, Any]] = []
	id_indices = split_indices[in_distribution_split]
	for dataset_index in id_indices:
		image, _ = dataset[dataset_index]
		# Saved MiraBest samples use trusted classifier preprocessing, as in calibration.
		tensor = deterministic_transform(image).unsqueeze(0)
		result = detector.validate_tensor(tensor)
		id_records.append(_record("in_distribution", str(dataset_index), result))

	ood_records = scan_ood_images(ood_test_dir, detector)

	records = id_records + ood_records
	paths = _paths(output_root)
	paths["output_dir"].mkdir(parents=True, exist_ok=True)
	_save_records(paths["csv"], records)
	id_rejected = sum(not record["accepted"] for record in id_records)
	ood_rejected = sum(not record["accepted"] for record in ood_records)

	summary = {
		"checkpoint": str(CHECKPOINT_PATH),
		"in_distribution_split": in_distribution_split,
		"threshold_calibration": {
			"feature_statistics_split": "train",
			"threshold_split": "validation",
			**threshold_dict,
		},
		"in_distribution": {
			"total": len(id_records),
			"accepted": len(id_records) - id_rejected,
			"rejected": id_rejected,
			"acceptance_rate": (len(id_records) - id_rejected) / len(id_records) if id_records else 0.0,
			"false_rejection_rate": id_rejected / len(id_records) if id_records else 0.0,
		},
		"ood": {
			"total": len(ood_records),
			"rejected": ood_rejected,
			"incorrectly_accepted": len(ood_records) - ood_rejected,
			"rejection_rate": ood_rejected / len(ood_records) if ood_records else None,
			"false_acceptance_rate": (len(ood_records) - ood_rejected) / len(ood_records) if ood_records else None,
		},
		"calibration_artifact": str(CALIBRATION_PATH),
		"ood_detection": {field: score_metrics(records, field)
			for field in ("energy_score", "feature_distance")},
		"limitation": "No finite OOD detector can guarantee rejection of every possible unseen input.",
	}
	paths["json"].write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
	_plot_scores(records, paths["figure"])
	return summary


def main() -> int:
	parser = argparse.ArgumentParser(description="Evaluate the ResNet input-domain/OOD safety layer.")
	parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
	parser.add_argument("--ood-test-dir", type=Path, default=PROJECT_ROOT / "data" / "ood" / "test")
	parser.add_argument("--in-distribution-split", choices=("validation", "test"), default="test")
	args = parser.parse_args()
	summary = run_ood_evaluation(args.config, PROJECT_ROOT / "outputs", args.ood_test_dir, args.in_distribution_split)
	print(json.dumps(summary, indent=2))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
