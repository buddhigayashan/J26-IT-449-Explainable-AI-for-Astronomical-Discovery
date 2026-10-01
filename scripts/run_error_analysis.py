"""Descriptive post-hoc error analysis for the locked ResNet test evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from PIL import Image
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.data.download_mirabest import load_mirabest
from src.data.preprocess import PreprocessConfig, build_transforms
from src.data.split_dataset import TransformedSubset
from src.inference.predict import CHECKPOINT_PATH, load_checkpoint, preprocess_image
from src.models.evaluate import get_test_indices


CLASS_NAMES = {0: "FRI", 1: "FRII"}
ERROR_TYPES = {
	(0, 0): "correct_FRI",
	(1, 1): "correct_FRII",
	(0, 1): "FRI_to_FRII",
	(1, 0): "FRII_to_FRI",
}
CSV_FIELDS = (
	"dataset_index",
	"true_label",
	"predicted_label",
	"probability_fri",
	"probability_frii",
	"predicted_confidence",
	"correct",
	"error_type",
)


def output_paths(output_root: str | Path = "outputs") -> dict[str, Path]:
	"""Return isolated error-analysis paths."""

	output_root = Path(output_root)
	analysis_dir = output_root / "error_analysis"
	return {
		"analysis_dir": analysis_dir,
		"predictions": analysis_dir / "test_predictions.csv",
		"summary": analysis_dir / "error_summary.json",
		"misclassified": analysis_dir / "misclassified_samples.csv",
		"high_confidence_errors": analysis_dir / "high_confidence_errors.csv",
		"low_confidence": analysis_dir / "low_confidence_samples.csv",
		"misclassified_figure": output_root / "figures" / "error_analysis_misclassified.png",
		"high_confidence_figure": output_root / "figures" / "error_analysis_high_confidence_errors.png",
		"low_confidence_figure": output_root / "figures" / "error_analysis_low_confidence.png",
	}


def load_test_indices(split_path: Path) -> list[int]:
	"""Load only the saved test index list; do not create a new split."""

	split_indices = json.loads(split_path.read_text(encoding="utf-8"))
	return get_test_indices(split_indices)


def build_inference_transform():
	"""Build the deterministic test transform with augmentation disabled."""

	_, evaluation_transform = build_transforms(
		PreprocessConfig(
			normalize_mean=0.5,
			normalize_std=0.5,
			rotation_degrees=0.0,
			horizontal_flip=False,
			vertical_flip=False,
		)
	)
	return evaluation_transform


def prediction_record(
	dataset_index: int,
	true_label: int,
	probabilities: list[float] | tuple[float, float],
) -> dict[str, Any]:
	predicted_label = int(probabilities[1] >= probabilities[0])
	confidence = float(probabilities[predicted_label])
	return {
		"dataset_index": dataset_index,
		"true_label": CLASS_NAMES[true_label],
		"predicted_label": CLASS_NAMES[predicted_label],
		"probability_fri": float(probabilities[0]),
		"probability_frii": float(probabilities[1]),
		"predicted_confidence": confidence,
		"correct": predicted_label == true_label,
		"error_type": ERROR_TYPES[(true_label, predicted_label)],
	}


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
	with path.open("w", newline="", encoding="utf-8") as file_handle:
		writer = csv.DictWriter(file_handle, fieldnames=list(CSV_FIELDS))
		writer.writeheader()
		writer.writerows({field: row[field] for field in CSV_FIELDS} for row in rows)


def _save_grid(
	samples: list[tuple[Image.Image, dict[str, Any]]], path: Path, title: str
) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	if not samples:
		figure, axis = plt.subplots(figsize=(6, 3))
		axis.axis("off")
		axis.set_title(f"{title}\nNo matching samples")
		figure.savefig(path, dpi=150, bbox_inches="tight")
		plt.close(figure)
		return

	columns = min(4, len(samples))
	rows_count = (len(samples) + columns - 1) // columns
	figure, axes = plt.subplots(rows_count, columns, figsize=(columns * 3, rows_count * 3))
	axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]
	for axis, (image, record) in zip(axes_list, samples):
		axis.imshow(image, cmap="gray")
		axis.set_title(
			f"True: {record['true_label']}\n"
			f"Pred: {record['predicted_label']}\n"
			f"Confidence: {record['predicted_confidence'] * 100:.1f}%"
		)
		axis.axis("off")
	for axis in axes_list[len(samples) :]:
		axis.axis("off")
	figure.suptitle(title)
	figure.tight_layout()
	figure.savefig(path, dpi=150)
	plt.close(figure)


def run_error_analysis(config_path: Path, output_root: Path) -> dict[str, Any]:
	"""Run descriptive inference on the fixed test split only."""

	config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
	dataset = load_mirabest(
		PROJECT_ROOT / config["paths"]["mirabest_root"], download=False
	)
	test_indices = load_test_indices(
		PROJECT_ROOT / config["paths"]["splits"] / "split_indices.json"
	)
	evaluation_transform = build_inference_transform()
	# Keep the transformed view explicit to document that only test indices are used.
	test_view = TransformedSubset(dataset, test_indices, evaluation_transform)
	model = load_checkpoint(str(CHECKPOINT_PATH))
	model.eval()
	rows: list[dict[str, Any]] = []
	sample_images: dict[int, Image.Image] = {}
	with torch.no_grad():
		for view_index, dataset_index in enumerate(test_indices):
			original_image, true_label = dataset[dataset_index]
			_, label_from_view = test_view[view_index]
			assert int(true_label) == int(label_from_view)
			tensor = preprocess_image(original_image)
			probabilities = torch.softmax(model(tensor), dim=1)[0].tolist()
			rows.append(prediction_record(dataset_index, int(true_label), probabilities))
			sample_images[dataset_index] = original_image.convert("L")

	paths = output_paths(output_root)
	paths["analysis_dir"].mkdir(parents=True, exist_ok=True)
	_write_rows(paths["predictions"], rows)
	misclassified = [row for row in rows if not row["correct"]]
	high_confidence_errors = [
		row for row in misclassified if row["predicted_confidence"] >= 0.80
	]
	low_confidence = [row for row in rows if row["predicted_confidence"] < 0.60]
	_write_rows(paths["misclassified"], misclassified)
	_write_rows(paths["high_confidence_errors"], high_confidence_errors)
	_write_rows(paths["low_confidence"], low_confidence)

	correct = [row for row in rows if row["correct"]]
	incorrect = misclassified
	summary = {
		"total_test_samples": len(rows),
		"total_correct": len(correct),
		"total_incorrect": len(incorrect),
		"FRI_to_FRII": sum(row["error_type"] == "FRI_to_FRII" for row in rows),
		"FRII_to_FRI": sum(row["error_type"] == "FRII_to_FRI" for row in rows),
		"mean_confidence_correct": (
			sum(row["predicted_confidence"] for row in correct) / len(correct)
			if correct
			else 0.0
		),
		"mean_confidence_incorrect": (
			sum(row["predicted_confidence"] for row in incorrect) / len(incorrect)
			if incorrect
			else 0.0
		),
		"high_confidence_error_count": len(high_confidence_errors),
		"low_confidence_count": len(low_confidence),
		"checkpoint": str(CHECKPOINT_PATH),
		"test_split_evaluated": True,
	}
	paths["summary"].write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

	for records, figure_path, title in (
		(misclassified, paths["misclassified_figure"], "Misclassified test samples"),
		(high_confidence_errors, paths["high_confidence_figure"], "High-confidence errors"),
		(low_confidence, paths["low_confidence_figure"], "Low-confidence test samples"),
	):
		_save_grid(
			[(sample_images[row["dataset_index"]], row) for row in records],
			figure_path,
			title,
		)
	return summary


def main() -> int:
	parser = argparse.ArgumentParser(description="Run descriptive ResNet test error analysis.")
	parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
	parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs")
	args = parser.parse_args()
	summary = run_error_analysis(args.config, args.output_root)
	print("ResNet test error analysis complete")
	print(json.dumps(summary, indent=2))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())