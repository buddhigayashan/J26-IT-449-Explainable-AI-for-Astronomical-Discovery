"""Validation-only evaluation and comparison for trained model variants."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
	accuracy_score,
	balanced_accuracy_score,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset


LABELS = (0, 1)
CLASS_NAMES = ("FRI", "FRII")
METRIC_KEYS = {
	"num_validation_samples",
	"validation_loss",
	"accuracy",
	"precision_fri",
	"precision_frii",
	"recall_fri",
	"recall_frii",
	"f1_fri",
	"f1_frii",
	"macro_precision",
	"macro_recall",
	"macro_f1",
	"weighted_f1",
	"balanced_accuracy",
	"confusion_matrix",
}


def get_validation_indices(split_indices: dict[str, list[int]]) -> list[int]:
	"""Return validation indices without accessing the test split."""

	return list(split_indices["validation"])


def evaluate_model(
	model: nn.Module,
	checkpoint_path: str | Path,
	validation_dataset: Dataset,
	batch_size: int = 32,
	device: str = "cpu",
) -> dict[str, Any]:
	"""Evaluate one checkpoint on the supplied validation dataset only."""

	device_object = torch.device(device)
	if device_object.type == "cuda" and not torch.cuda.is_available():
		raise RuntimeError("CUDA was requested but is unavailable.")
	checkpoint = torch.load(checkpoint_path, map_location=device_object, weights_only=False)
	state_dict = checkpoint.get("model_state_dict", checkpoint)
	model.load_state_dict(state_dict)
	model.to(device_object)
	model.eval()
	loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
	criterion = nn.CrossEntropyLoss()
	y_true: list[int] = []
	y_predicted: list[int] = []
	total_loss = 0.0
	total_samples = 0
	with torch.no_grad():
		for images, labels in loader:
			images = images.to(device_object)
			labels = labels.to(device_object)
			logits = model(images)
			loss = criterion(logits, labels)
			predictions = logits.argmax(dim=1)
			total_loss += loss.item() * labels.size(0)
			total_samples += labels.size(0)
			y_true.extend(labels.cpu().tolist())
			y_predicted.extend(predictions.cpu().tolist())

	if total_samples == 0:
		raise ValueError("Cannot evaluate an empty validation dataset.")
	confusion = confusion_matrix(y_true, y_predicted, labels=LABELS)
	return {
		"num_validation_samples": total_samples,
		"validation_loss": total_loss / total_samples,
		"accuracy": accuracy_score(y_true, y_predicted),
		"precision_fri": precision_score(y_true, y_predicted, labels=LABELS, pos_label=0, zero_division=0),
		"precision_frii": precision_score(y_true, y_predicted, labels=LABELS, pos_label=1, zero_division=0),
		"recall_fri": recall_score(y_true, y_predicted, labels=LABELS, pos_label=0, zero_division=0),
		"recall_frii": recall_score(y_true, y_predicted, labels=LABELS, pos_label=1, zero_division=0),
		"f1_fri": f1_score(y_true, y_predicted, labels=LABELS, pos_label=0, zero_division=0),
		"f1_frii": f1_score(y_true, y_predicted, labels=LABELS, pos_label=1, zero_division=0),
		"macro_precision": precision_score(y_true, y_predicted, labels=LABELS, average="macro", zero_division=0),
		"macro_recall": recall_score(y_true, y_predicted, labels=LABELS, average="macro", zero_division=0),
		"macro_f1": f1_score(y_true, y_predicted, labels=LABELS, average="macro", zero_division=0),
		"weighted_f1": f1_score(y_true, y_predicted, labels=LABELS, average="weighted", zero_division=0),
		"balanced_accuracy": balanced_accuracy_score(y_true, y_predicted),
		"confusion_matrix": confusion.tolist(),
	}


def _save_confusion_matrix(
	confusion: list[list[int]], model_name: str, output_path: Path
) -> None:
	figure, axis = plt.subplots(figsize=(5, 4))
	image = axis.imshow(np.asarray(confusion), cmap="Blues")
	figure.colorbar(image, ax=axis)
	axis.set_xlabel("Predicted label")
	axis.set_ylabel("True label")
	axis.set_xticks(range(2), CLASS_NAMES)
	axis.set_yticks(range(2), CLASS_NAMES)
	axis.set_title(f"{model_name.title()} validation confusion matrix")
	for row in range(2):
		for column in range(2):
			axis.text(column, row, confusion[row][column], ha="center", va="center")
	figure.tight_layout()
	figure.savefig(output_path, dpi=150)
	plt.close(figure)


def save_comparison_results(
	results: dict[str, dict[str, Any]], output_root: str | Path = "outputs"
) -> tuple[dict[str, Path], Path, Path]:
	"""Save per-model JSON, comparison CSV, confusion matrices, and summary plot."""

	output_root = Path(output_root)
	metrics_dir = output_root / "metrics"
	confusion_dir = output_root / "confusion_matrix"
	figures_dir = output_root / "figures"
	for directory in (metrics_dir, confusion_dir, figures_dir):
		directory.mkdir(parents=True, exist_ok=True)

	metric_paths: dict[str, Path] = {}
	for model_name, metrics in results.items():
		path = metrics_dir / f"{model_name}_validation_evaluation.json"
		path.write_text(
			json.dumps({"model": model_name, "test_evaluation_performed": False, **metrics}, indent=2)
			+ "\n",
			encoding="utf-8",
		)
		metric_paths[model_name] = path
		_save_confusion_matrix(
			metrics["confusion_matrix"],
			model_name,
			confusion_dir / f"{model_name}_validation_confusion_matrix.png",
		)

	comparison_fields = [
		"model",
		"validation_loss",
		"accuracy",
		"precision_fri",
		"precision_frii",
		"recall_fri",
		"recall_frii",
		"f1_fri",
		"f1_frii",
		"macro_precision",
		"macro_recall",
		"macro_f1",
		"weighted_f1",
		"balanced_accuracy",
	]
	comparison_path = metrics_dir / "model_validation_comparison.csv"
	with comparison_path.open("w", newline="", encoding="utf-8") as file_handle:
		writer = csv.DictWriter(file_handle, fieldnames=comparison_fields)
		writer.writeheader()
		for model_name, metrics in results.items():
			writer.writerow({field: model_name if field == "model" else metrics[field] for field in comparison_fields})

	figure, axis = plt.subplots(figsize=(8, 5))
	model_names = list(results)
	positions = np.arange(len(model_names))
	width = 0.25
	for offset, metric_name in enumerate(("accuracy", "macro_f1", "balanced_accuracy")):
		axis.bar(
			positions + (offset - 1) * width,
			[results[name][metric_name] for name in model_names],
			width,
			label=metric_name.replace("_", " ").title(),
		)
	axis.set_xticks(positions, [name.title() for name in model_names])
	axis.set_ylim(0, 1)
	axis.set_ylabel("Score")
	axis.set_title("Validation model comparison")
	axis.legend()
	figure.tight_layout()
	comparison_figure_path = figures_dir / "model_validation_comparison.png"
	figure.savefig(comparison_figure_path, dpi=150)
	plt.close(figure)
	return metric_paths, comparison_path, comparison_figure_path
