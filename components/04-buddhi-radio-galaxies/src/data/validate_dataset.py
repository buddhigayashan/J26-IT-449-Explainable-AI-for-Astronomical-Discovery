"""Inspection and validation for the MiraBest Zenodo v1 binary dataset."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


LABEL_MAPPING = {0: "FR-I", 1: "FR-II"}


def _image_to_tensor(image: Any) -> torch.Tensor:
	if isinstance(image, Image.Image):
		array = np.asarray(image)
		tensor = torch.from_numpy(array.copy())
	elif torch.is_tensor(image):
		tensor = image.detach().cpu()
	else:
		tensor = torch.as_tensor(np.asarray(image))

	if tensor.ndim == 2:
		tensor = tensor.unsqueeze(0)
	elif tensor.ndim == 3 and tensor.shape[-1] == 1:
		tensor = tensor.permute(2, 0, 1)
	if tensor.ndim != 3:
		raise ValueError(f"Expected a grayscale image with 2 or 3 dimensions, got {tuple(tensor.shape)}")
	return tensor


def validate_dataset(dataset: Any) -> dict[str, Any]:
	"""Calculate validation statistics from every loaded sample."""

	class_counts = {label: 0 for label in LABEL_MAPPING}
	unreadable_samples: list[int] = []
	invalid_labels: dict[str, int] = {}
	nan_count = 0
	inf_count = 0
	pixel_min: float | None = None
	pixel_max: float | None = None
	image_shapes: set[tuple[int, ...]] = set()
	source_dtypes: set[str] = set()
	valid_samples = 0

	for index in range(len(dataset)):
		try:
			image, label = dataset[index]
			label = int(label)
			if label not in LABEL_MAPPING:
				invalid_labels[str(label)] = invalid_labels.get(str(label), 0) + 1
				continue

			tensor = _image_to_tensor(image)
			image_shapes.add(tuple(int(value) for value in tensor.shape))
			source_dtypes.add(str(tensor.dtype).replace("torch.", ""))
			numeric_tensor = tensor.to(torch.float64)
			nan_count += int(torch.isnan(numeric_tensor).sum().item())
			inf_count += int(torch.isinf(numeric_tensor).sum().item())
			sample_min = float(torch.min(numeric_tensor).item())
			sample_max = float(torch.max(numeric_tensor).item())
			pixel_min = sample_min if pixel_min is None else min(pixel_min, sample_min)
			pixel_max = sample_max if pixel_max is None else max(pixel_max, sample_max)
			class_counts[label] += 1
			valid_samples += 1
		except (OSError, ValueError, TypeError, RuntimeError, IndexError):
			unreadable_samples.append(index)

	total_samples = len(dataset)
	percentages = {
		LABEL_MAPPING[label]: round((count / total_samples) * 100, 6)
		if total_samples
		else 0.0
		for label, count in class_counts.items()
	}
	common_shape = list(next(iter(image_shapes))) if len(image_shapes) == 1 else None
	dimensions = list(common_shape[-2:]) if common_shape is not None else None

	return {
		"dataset": "MiraBest Zenodo v1 basic binary dataset",
		"source": "https://github.com/as595/E2CNNRadGal/blob/main/MiraBest.py",
		"total_samples": total_samples,
		"valid_samples": valid_samples,
		"class_counts": {LABEL_MAPPING[label]: count for label, count in class_counts.items()},
		"class_balance_percentages": percentages,
		"label_values": sorted(class_counts),
		"label_mapping": {str(label): name for label, name in LABEL_MAPPING.items()},
		"image_tensor_shape": common_shape,
		"image_dimensions": dimensions,
		"dtype": sorted(source_dtypes),
		"minimum_pixel_value": pixel_min,
		"maximum_pixel_value": pixel_max,
		"nan_values": nan_count,
		"inf_values": inf_count,
		"unreadable_sample_count": len(unreadable_samples),
		"unreadable_sample_indices": unreadable_samples,
		"invalid_labels": invalid_labels,
		"preprocessing_applied": False,
	}


def save_validation_results(
	results: dict[str, Any], metrics_dir: str | Path = "outputs/metrics"
) -> tuple[Path, Path]:
	"""Save validation results as JSON and class distribution CSV."""

	metrics_dir = Path(metrics_dir)
	metrics_dir.mkdir(parents=True, exist_ok=True)
	json_path = metrics_dir / "dataset_validation.json"
	csv_path = metrics_dir / "class_distribution.csv"
	json_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

	with csv_path.open("w", newline="", encoding="utf-8") as file_handle:
		writer = csv.DictWriter(file_handle, fieldnames=["label", "class", "count", "percentage"])
		writer.writeheader()
		for label, class_name in LABEL_MAPPING.items():
			writer.writerow(
				{
					"label": label,
					"class": class_name,
					"count": results["class_counts"][class_name],
					"percentage": results["class_balance_percentages"][class_name],
				}
			)
	return json_path, csv_path


def save_sample_grid(
	dataset: Any,
	output_path: str | Path = "outputs/figures/mirabest_sample_grid.png",
	samples_per_class: int = 5,
) -> Path:
	"""Save real grayscale examples from both binary classes without transforms."""

	selected: dict[int, list[Image.Image]] = {label: [] for label in LABEL_MAPPING}
	for index in range(len(dataset)):
		image, label = dataset[index]
		label = int(label)
		if label in selected and len(selected[label]) < samples_per_class:
			if not isinstance(image, Image.Image):
				tensor = _image_to_tensor(image)
				image = Image.fromarray(tensor.squeeze(0).numpy())
			selected[label].append(image.convert("L"))
		if all(len(images) == samples_per_class for images in selected.values()):
			break

	if any(not images for images in selected.values()):
		raise ValueError("Both FR-I and FR-II samples are required for the sample grid.")

	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	figure, axes = plt.subplots(2, samples_per_class, figsize=(samples_per_class * 2.2, 4.8))
	axes = np.asarray(axes).reshape(2, samples_per_class)
	for row, label in enumerate(LABEL_MAPPING):
		for column, image in enumerate(selected[label]):
			axes[row, column].imshow(image, cmap="gray")
			axes[row, column].set_title(LABEL_MAPPING[label])
			axes[row, column].axis("off")
	figure.suptitle("MiraBest Zenodo v1 basic binary samples")
	figure.tight_layout()
	figure.savefig(output_path, dpi=150)
	plt.close(figure)
	return output_path
