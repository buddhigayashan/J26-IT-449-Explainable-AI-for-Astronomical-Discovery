"""Reproducible stratified splitting for the validated MiraBest dataset."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from src.data.preprocess import PreprocessConfig, build_transforms


LABEL_NAMES = {0: "FR-I", 1: "FR-II"}
DEFAULT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}


class TransformedSubset(Dataset):
	"""A dataset view with a transform applied only when a sample is read."""

	def __init__(self, dataset: Dataset, indices: Sequence[int], transform: Any) -> None:
		self.dataset = dataset
		self.indices = list(indices)
		self.transform = transform

	def __len__(self) -> int:
		return len(self.indices)

	def __getitem__(self, index: int) -> tuple[Any, int]:
		image, label = self.dataset[self.indices[index]]
		return self.transform(image), int(label)


def stratified_split_indices(
	labels: Sequence[int],
	seed: int,
	ratios: dict[str, float] | None = None,
) -> dict[str, list[int]]:
	"""Create disjoint, reproducible train/validation/test index lists."""

	ratios = ratios or DEFAULT_RATIOS
	if set(ratios) != set(DEFAULT_RATIOS) or abs(sum(ratios.values()) - 1.0) > 1e-9:
		raise ValueError("Split ratios must contain train, validation, test and sum to 1.")
	if len(labels) < 3:
		raise ValueError("At least three samples are required for splitting.")

	indices = list(range(len(labels)))
	train_validation, test = train_test_split(
		indices,
		test_size=ratios["test"],
		stratify=list(labels),
		random_state=seed,
	)
	validation_fraction = ratios["validation"] / (ratios["train"] + ratios["validation"])
	train, validation = train_test_split(
		train_validation,
		test_size=validation_fraction,
		stratify=[labels[index] for index in train_validation],
		random_state=seed,
	)

	return {
		"train": sorted(train),
		"validation": sorted(validation),
		"test": sorted(test),
	}


def _split_counts(dataset: Dataset, indices: Sequence[int]) -> Counter:
	return Counter(int(dataset[index][1]) for index in indices)


def _class_distribution_rows(dataset: Dataset, split_indices: dict[str, list[int]]) -> list[dict[str, Any]]:
	rows = []
	for split_name, indices in split_indices.items():
		counts = _split_counts(dataset, indices)
		for label, class_name in LABEL_NAMES.items():
			rows.append(
				{
					"split": split_name,
					"label": label,
					"class": class_name,
					"count": counts[label],
					"percentage": round((counts[label] / len(indices)) * 100, 6),
				}
			)
	return rows


def save_split_artifacts(
	dataset: Dataset,
	split_indices: dict[str, list[int]],
	output_dir: str | Path = "data/splits",
	figure_path: str | Path = "outputs/figures/split_class_distribution.png",
	seed: int = 42,
	ratios: dict[str, float] | None = None,
) -> tuple[Path, Path, Path]:
	"""Save reproducible indices, summary JSON/CSV, and a distribution figure."""

	ratios = ratios or DEFAULT_RATIOS
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	figure_path = Path(figure_path)
	figure_path.parent.mkdir(parents=True, exist_ok=True)

	index_path = output_dir / "split_indices.json"
	summary_path = output_dir / "split_summary.json"
	csv_path = output_dir / "split_class_distribution.csv"
	index_path.write_text(json.dumps(split_indices, indent=2) + "\n", encoding="utf-8")

	rows = _class_distribution_rows(dataset, split_indices)
	summary = {
		"dataset": "MiraBest Zenodo v1 basic binary dataset",
		"seed": seed,
		"requested_ratios": ratios,
		"total_samples": len(dataset),
		"splits": {
			split: {
				"count": len(indices),
				"proportion": round(len(indices) / len(dataset), 6),
				"class_counts": {
					LABEL_NAMES[label]: _split_counts(dataset, indices)[label]
					for label in LABEL_NAMES
				},
			}
			for split, indices in split_indices.items()
		},
		"preprocessing": {
			"image_size_preserved": True,
			"train_augmentation_only": True,
			"raw_data_modified": False,
		},
	}
	summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

	with csv_path.open("w", newline="", encoding="utf-8") as file_handle:
		writer = csv.DictWriter(file_handle, fieldnames=["split", "label", "class", "count", "percentage"])
		writer.writeheader()
		writer.writerows(rows)

	figure, axis = plt.subplots(figsize=(7, 4.5))
	split_names = list(split_indices)
	positions = list(range(len(split_names)))
	width = 0.35
	for offset, label in enumerate(LABEL_NAMES):
		counts = [_split_counts(dataset, split_indices[split])[label] for split in split_names]
		axis.bar(
			[position + (offset - 0.5) * width for position in positions],
			counts,
			width,
			label=LABEL_NAMES[label],
		)
	axis.set_xticks(positions, [name.title() for name in split_names])
	axis.set_ylabel("Samples")
	axis.set_title("MiraBest Zenodo v1 class distribution by split")
	axis.legend()
	figure.tight_layout()
	figure.savefig(figure_path, dpi=150)
	plt.close(figure)
	return index_path, summary_path, csv_path


def build_preprocessed_splits(
	dataset: Dataset,
	split_indices: dict[str, list[int]],
	config: PreprocessConfig | None = None,
) -> dict[str, TransformedSubset]:
	"""Build transformed views with augmentation restricted to training."""

	train_transform, evaluation_transform = build_transforms(config)
	return {
		"train": TransformedSubset(dataset, split_indices["train"], train_transform),
		"validation": TransformedSubset(dataset, split_indices["validation"], evaluation_transform),
		"test": TransformedSubset(dataset, split_indices["test"], evaluation_transform),
	}


def prepare_splits(
	dataset: Dataset,
	seed: int,
	output_dir: str | Path = "data/splits",
	figure_path: str | Path = "outputs/figures/split_class_distribution.png",
	ratios: dict[str, float] | None = None,
	preprocess_config: PreprocessConfig | None = None,
) -> tuple[dict[str, list[int]], dict[str, TransformedSubset]]:
	"""Create, persist, and return reproducible preprocessed dataset views."""

	labels = [int(dataset[index][1]) for index in range(len(dataset))]
	split_indices = stratified_split_indices(labels, seed=seed, ratios=ratios)
	save_split_artifacts(dataset, split_indices, output_dir, figure_path, seed, ratios)
	return split_indices, build_preprocessed_splits(dataset, split_indices, preprocess_config)


def main() -> int:
	parser = argparse.ArgumentParser(description="Create MiraBest preprocessing and stratified split artifacts.")
	parser.add_argument("--config", type=Path, default=Path("config.yaml"))
	args = parser.parse_args()

	import yaml

	config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
	project_root = Path(__file__).resolve().parents[2]
	if str(project_root) not in sys.path:
		sys.path.insert(0, str(project_root))
	from src.data.download_mirabest import load_mirabest

	dataset_root = project_root / config["paths"]["mirabest_root"]
	dataset = load_mirabest(dataset_root, download=False)
	seed = int(config["training"]["seed"])
	split_config = config["splitting"]
	ratios = {
		"train": float(split_config["train_ratio"]),
		"validation": float(split_config["validation_ratio"]),
		"test": float(split_config["test_ratio"]),
	}
	preprocessing_config = config["preprocessing"]
	preprocess_config = PreprocessConfig(
		normalize_mean=float(preprocessing_config["normalize_mean"]),
		normalize_std=float(preprocessing_config["normalize_std"]),
		rotation_degrees=float(preprocessing_config["rotation_degrees"]),
		horizontal_flip=bool(preprocessing_config["horizontal_flip"]),
		vertical_flip=bool(preprocessing_config["vertical_flip"]),
	)
	split_indices, _ = prepare_splits(
		dataset,
		seed=seed,
		output_dir=project_root / config["paths"]["splits"],
		figure_path=project_root / config["paths"]["figures"] / "split_class_distribution.png",
		ratios=ratios,
		preprocess_config=preprocess_config,
	)
	print("MiraBest Zenodo v1 preprocessing and stratified splitting")
	for split_name, indices in split_indices.items():
		print(f"{split_name.title()}: {len(indices)} samples")
	print(f"Seed: {seed}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
