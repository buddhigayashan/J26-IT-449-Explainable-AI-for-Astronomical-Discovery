"""Run reproducibility experiments for the selected ResNet on validation data."""

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.data.download_mirabest import load_mirabest
from src.data.preprocess import PreprocessConfig, build_transforms
from src.data.split_dataset import TransformedSubset
from src.models.evaluate import evaluate_model
from src.models.resnet_cnn import ResNetCNN
from src.models.train import TrainingConfig, train_model
from src.utils.seed import set_global_seed


SEEDS = (42, 7, 99, 123, 2026)
RESULT_FIELDS = (
	"seed",
	"best_epoch",
	"best_validation_loss",
	"validation_accuracy",
	"macro_precision",
	"macro_recall",
	"macro_f1",
	"balanced_accuracy",
)


def load_fixed_split_indices(split_path: Path) -> dict[str, list[int]]:
	"""Load existing indices without generating or changing a split."""

	if not split_path.is_file():
		raise FileNotFoundError(f"Saved split indices not found: {split_path}")
	split_indices = json.loads(split_path.read_text(encoding="utf-8"))
	if set(split_indices) != {"train", "validation", "test"}:
		raise ValueError("Saved split indices must contain train, validation, and test keys.")
	return split_indices


def seed_artifact_paths(output_root: str | Path, seed: int) -> dict[str, Path]:
	"""Return isolated paths for one seed without touching the final checkpoint."""

	output_root = Path(output_root)
	name = f"resnet_seed_{seed}"
	reproducibility_dir = output_root / "reproducibility"
	return {
		"checkpoint": output_root / "models" / "reproducibility" / f"{name}_best.pth",
		"history": reproducibility_dir / f"{name}_training_history.csv",
		"curves": reproducibility_dir / f"{name}_training_curves.png",
		"validation_metrics": reproducibility_dir / f"{name}_metrics.json",
	}


def build_seed_views(dataset: Any, split_indices: dict[str, list[int]], config: PreprocessConfig):
	"""Build only train and validation views from the fixed indices."""

	train_transform, evaluation_transform = build_transforms(config)
	return (
		TransformedSubset(dataset, split_indices["train"], train_transform),
		TransformedSubset(dataset, split_indices["validation"], evaluation_transform),
	)


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, float]:
	"""Calculate population mean/std statistics across seed results."""

	metrics = {
		"validation_accuracy": "mean_validation_accuracy",
		"macro_f1": "mean_macro_f1",
		"balanced_accuracy": "mean_balanced_accuracy",
		"best_validation_loss": "mean_validation_loss",
	}
	summary: dict[str, float] = {}
	for field, mean_name in metrics.items():
		values = np.asarray([float(result[field]) for result in results], dtype=np.float64)
		summary[mean_name] = float(values.mean())
		std_name = mean_name.replace("mean_", "std_")
		summary[std_name] = float(values.std(ddof=0))
	return summary


def save_reproducibility_artifacts(
	results: list[dict[str, Any]], output_root: str | Path
) -> tuple[Path, Path, Path]:
	"""Save per-seed CSV, aggregate JSON, and stability figure."""

	output_root = Path(output_root)
	reproducibility_dir = output_root / "reproducibility"
	figures_dir = output_root / "figures"
	reproducibility_dir.mkdir(parents=True, exist_ok=True)
	figures_dir.mkdir(parents=True, exist_ok=True)
	results_csv = reproducibility_dir / "resnet_multiseed_results.csv"
	with results_csv.open("w", newline="", encoding="utf-8") as file_handle:
		writer = csv.DictWriter(file_handle, fieldnames=list(RESULT_FIELDS))
		writer.writeheader()
		writer.writerows({field: result[field] for field in RESULT_FIELDS} for result in results)

	results_summary = aggregate_results(results)
	summary_json = reproducibility_dir / "resnet_multiseed_summary.json"
	summary_json.write_text(
		json.dumps(
			{
				"model": "resnet",
				"seeds": [result["seed"] for result in results],
				"test_evaluation_performed": False,
				**results_summary,
			},
			indent=2,
		)
		+ "\n",
		encoding="utf-8",
	)

	figure, axis = plt.subplots(figsize=(8, 5))
	positions = np.arange(len(results))
	width = 0.25
	for offset, field in enumerate(("validation_accuracy", "macro_f1", "balanced_accuracy")):
		axis.bar(
			positions + (offset - 1) * width,
			[result[field] for result in results],
			width,
			label=field.replace("_", " ").title(),
		)
	axis.set_xticks(positions, [str(result["seed"]) for result in results])
	axis.set_xlabel("Random seed")
	axis.set_ylabel("Validation score")
	axis.set_ylim(0, 1)
	axis.set_title("ResNet validation stability across seeds")
	axis.legend()
	figure.tight_layout()
	figure_path = figures_dir / "resnet_multiseed_stability.png"
	figure.savefig(figure_path, dpi=150)
	plt.close(figure)
	return results_csv, summary_json, figure_path


def run_seed(
	seed: int,
	dataset: Any,
	split_indices: dict[str, list[int]],
	preprocess_config: PreprocessConfig,
	training_config: TrainingConfig,
	output_root: Path,
) -> dict[str, Any]:
	"""Train and validate one seed using the fixed train/validation indices."""

	train_dataset, validation_dataset = build_seed_views(dataset, split_indices, preprocess_config)
	paths = seed_artifact_paths(output_root, seed)
	set_global_seed(seed)
	seed_config = TrainingConfig(
		seed=seed,
		batch_size=training_config.batch_size,
		learning_rate=training_config.learning_rate,
		weight_decay=training_config.weight_decay,
		num_epochs=training_config.num_epochs,
		num_workers=training_config.num_workers,
		early_stopping_patience=training_config.early_stopping_patience,
		scheduler_patience=training_config.scheduler_patience,
		scheduler_factor=training_config.scheduler_factor,
		device=training_config.device,
	)
	training_result = train_model(
		train_dataset,
		validation_dataset,
		config=seed_config,
		model=ResNetCNN(),
		output_root=output_root,
		artifact_prefix=f"resnet_seed_{seed}",
		artifact_paths=paths,
	)
	validation_metrics = evaluate_model(
		ResNetCNN(),
		paths["checkpoint"],
		validation_dataset,
		batch_size=seed_config.batch_size,
		device=seed_config.device,
		split_name="validation",
	)
	result = {
		"seed": seed,
		"best_epoch": training_result["best_epoch"],
		"best_validation_loss": training_result["best_validation_loss"],
		"validation_accuracy": validation_metrics["accuracy"],
		"macro_precision": validation_metrics["macro_precision"],
		"macro_recall": validation_metrics["macro_recall"],
		"macro_f1": validation_metrics["macro_f1"],
		"balanced_accuracy": validation_metrics["balanced_accuracy"],
		"test_evaluation_performed": False,
	}
	paths["validation_metrics"].write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
	return result


def main() -> int:
	parser = argparse.ArgumentParser(description="Run the five-seed ResNet reproducibility experiment.")
	parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
	parser.add_argument(
		"--seed",
		type=int,
		default=None,
		help="Run one seed only for isolated verification instead of all five seeds.",
	)
	parser.add_argument(
		"--output-root",
		type=Path,
		default=PROJECT_ROOT / "outputs",
		help="Root for reproducibility artifacts.",
	)
	args = parser.parse_args()
	config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
	split_indices = load_fixed_split_indices(
		PROJECT_ROOT / config["paths"]["splits"] / "split_indices.json"
	)
	dataset = load_mirabest(PROJECT_ROOT / config["paths"]["mirabest_root"], download=False)
	preprocessing = config["preprocessing"]
	preprocess_config = PreprocessConfig(
		normalize_mean=float(preprocessing["normalize_mean"]),
		normalize_std=float(preprocessing["normalize_std"]),
		rotation_degrees=float(preprocessing["rotation_degrees"]),
		horizontal_flip=bool(preprocessing["horizontal_flip"]),
		vertical_flip=bool(preprocessing["vertical_flip"]),
	)
	base_training = config["training"]
	resnet_config = config["models"]["resnet"]
	training_config = TrainingConfig(
		seed=42,
		batch_size=int(base_training["batch_size"]),
		learning_rate=float(resnet_config["learning_rate"]),
		weight_decay=float(resnet_config["weight_decay"]),
		num_epochs=int(resnet_config["num_epochs"]),
		num_workers=int(base_training["num_workers"]),
		early_stopping_patience=int(resnet_config["early_stopping_patience"]),
		scheduler_patience=int(resnet_config["scheduler_patience"]),
		scheduler_factor=float(resnet_config["scheduler_factor"]),
		device=str(base_training["device"]),
	)
	selected_seeds = (args.seed,) if args.seed is not None else SEEDS
	results = [
		run_seed(seed, dataset, split_indices, preprocess_config, training_config, args.output_root)
		for seed in selected_seeds
	]
	results_csv, summary_json, figure_path = save_reproducibility_artifacts(
		results, args.output_root
	)
	print("ResNet multiseed reproducibility experiment complete")
	print(f"Seeds: {', '.join(str(seed) for seed in selected_seeds)}")
	print(f"Results CSV: {results_csv}")
	print(f"Summary JSON: {summary_json}")
	print(f"Stability figure: {figure_path}")
	print("Test evaluation: not performed")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())