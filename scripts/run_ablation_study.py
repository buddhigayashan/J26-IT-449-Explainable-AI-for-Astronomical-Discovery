"""Run controlled ResNet training ablations on train/validation data only."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
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


SEED = 42
ABLATION_NAMES = ("full_resnet", "no_augmentation", "no_weight_decay", "no_scheduler")
RESULT_FIELDS = (
	"configuration",
	"best_epoch",
	"best_validation_loss",
	"validation_accuracy",
	"macro_precision",
	"macro_recall",
	"macro_f1",
	"balanced_accuracy",
)


@dataclass(frozen=True)
class AblationSpec:
	name: str
	augmentation_enabled: bool
	weight_decay: float
	scheduler_enabled: bool


def get_ablation_specs(weight_decay: float = 0.0001) -> tuple[AblationSpec, ...]:
	"""Return the four controlled configurations relative to the full reference."""

	return (
		AblationSpec("full_resnet", True, weight_decay, True),
		AblationSpec("no_augmentation", False, weight_decay, True),
		AblationSpec("no_weight_decay", True, 0.0, True),
		AblationSpec("no_scheduler", True, weight_decay, False),
	)


def load_fixed_split_indices(split_path: Path) -> dict[str, list[int]]:
	if not split_path.is_file():
		raise FileNotFoundError(f"Saved split indices not found: {split_path}")
	split_indices = json.loads(split_path.read_text(encoding="utf-8"))
	if set(split_indices) != {"train", "validation", "test"}:
		raise ValueError("Saved split indices must contain train, validation, and test keys.")
	return split_indices


def ablation_artifact_paths(output_root: str | Path, name: str) -> dict[str, Path]:
	"""Return isolated artifact paths for an ablation configuration."""

	output_root = Path(output_root)
	return {
		"checkpoint": output_root / "models" / "ablation" / f"{name}_best.pth",
		"history": output_root / "ablation" / f"{name}_training_history.csv",
		"curves": output_root / "ablation" / f"{name}_training_curves.png",
		"validation_metrics": output_root / "ablation" / f"{name}_metrics.json",
	}


def build_ablation_views(
	dataset: Any,
	split_indices: dict[str, list[int]],
	preprocessing: PreprocessConfig,
	spec: AblationSpec,
) -> tuple[TransformedSubset, TransformedSubset]:
	"""Build train/validation views only; the test split is never accessed."""

	config = preprocessing
	if not spec.augmentation_enabled:
		config = PreprocessConfig(
			normalize_mean=preprocessing.normalize_mean,
			normalize_std=preprocessing.normalize_std,
			rotation_degrees=0.0,
			horizontal_flip=False,
			vertical_flip=False,
		)
	train_transform, evaluation_transform = build_transforms(config)
	return (
		TransformedSubset(dataset, split_indices["train"], train_transform),
		TransformedSubset(dataset, split_indices["validation"], evaluation_transform),
	)


def build_training_config(
	base_training: dict[str, Any],
	resnet_config: dict[str, Any],
	spec: AblationSpec,
) -> TrainingConfig:
	return TrainingConfig(
		seed=SEED,
		batch_size=int(base_training["batch_size"]),
		learning_rate=float(resnet_config["learning_rate"]),
		weight_decay=spec.weight_decay,
		num_epochs=int(resnet_config["num_epochs"]),
		num_workers=int(base_training["num_workers"]),
		early_stopping_patience=int(resnet_config["early_stopping_patience"]),
		scheduler_patience=(
		int(resnet_config["scheduler_patience"])
		if spec.scheduler_enabled
		else None
	),
		scheduler_factor=float(resnet_config["scheduler_factor"]),
		device=str(base_training["device"]),
	)


def run_ablation(
	spec: AblationSpec,
	dataset: Any,
	split_indices: dict[str, list[int]],
	preprocessing: PreprocessConfig,
	training_config: TrainingConfig,
	output_root: Path,
) -> dict[str, Any]:
	"""Run one deterministic ablation on train/validation data only."""

	set_global_seed(SEED)
	train_dataset, validation_dataset = build_ablation_views(
		dataset, split_indices, preprocessing, spec
	)
	paths = ablation_artifact_paths(output_root, spec.name)
	training_result = train_model(
		train_dataset,
		validation_dataset,
		config=training_config,
		model=ResNetCNN(),
		output_root=output_root,
		artifact_prefix=f"ablation_{spec.name}",
		artifact_paths=paths,
	)
	validation_metrics = evaluate_model(
		ResNetCNN(),
		paths["checkpoint"],
		validation_dataset,
		batch_size=training_config.batch_size,
		device=training_config.device,
		split_name="validation",
	)
	result = {
		"configuration": spec.name,
		"best_epoch": training_result["best_epoch"],
		"best_validation_loss": training_result["best_validation_loss"],
		"validation_accuracy": validation_metrics["accuracy"],
		"macro_precision": validation_metrics["macro_precision"],
		"macro_recall": validation_metrics["macro_recall"],
		"macro_f1": validation_metrics["macro_f1"],
		"balanced_accuracy": validation_metrics["balanced_accuracy"],
		"seed": SEED,
		"test_evaluation_performed": False,
	}
	paths["validation_metrics"].write_text(
		json.dumps(result, indent=2) + "\n", encoding="utf-8"
	)
	return result


def save_ablation_summary(results: list[dict[str, Any]], output_root: Path) -> tuple[Path, Path]:
	ablation_dir = output_root / "ablation"
	figures_dir = output_root / "figures"
	ablation_dir.mkdir(parents=True, exist_ok=True)
	figures_dir.mkdir(parents=True, exist_ok=True)
	results_path = ablation_dir / "ablation_results.csv"
	with results_path.open("w", newline="", encoding="utf-8") as file_handle:
		writer = csv.DictWriter(file_handle, fieldnames=list(RESULT_FIELDS))
		writer.writeheader()
		writer.writerows({field: result[field] for field in RESULT_FIELDS} for result in results)

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
	axis.set_xticks(positions, [result["configuration"].replace("_", " ").title() for result in results])
	axis.set_ylim(0, 1)
	axis.set_ylabel("Validation score")
	axis.set_title("ResNet ablation comparison")
	axis.legend()
	figure.tight_layout()
	figure_path = figures_dir / "ablation_comparison.png"
	figure.savefig(figure_path, dpi=150)
	plt.close(figure)
	return results_path, figure_path


def main() -> int:
	parser = argparse.ArgumentParser(description="Run the controlled ResNet ablation study.")
	parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
	parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs")
	args = parser.parse_args()
	config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
	split_indices = load_fixed_split_indices(
		PROJECT_ROOT / config["paths"]["splits"] / "split_indices.json"
	)
	dataset = load_mirabest(PROJECT_ROOT / config["paths"]["mirabest_root"], download=False)
	preprocessing_values = config["preprocessing"]
	preprocessing = PreprocessConfig(
		normalize_mean=float(preprocessing_values["normalize_mean"]),
		normalize_std=float(preprocessing_values["normalize_std"]),
		rotation_degrees=float(preprocessing_values["rotation_degrees"]),
		horizontal_flip=bool(preprocessing_values["horizontal_flip"]),
		vertical_flip=bool(preprocessing_values["vertical_flip"]),
	)
	base_training = config["training"]
	resnet_config = config["models"]["resnet"]
	specs = get_ablation_specs(float(resnet_config["weight_decay"]))
	results = []
	for spec in specs:
		training_config = build_training_config(base_training, resnet_config, spec)
		results.append(
			run_ablation(
				spec, dataset, split_indices, preprocessing, training_config, args.output_root
			)
		)
	results_path, figure_path = save_ablation_summary(results, args.output_root)
	print("ResNet ablation study complete")
	print(f"Configurations: {', '.join(ABLATION_NAMES)}")
	print(f"Results CSV: {results_path}")
	print(f"Comparison figure: {figure_path}")
	print("Test evaluation: not performed")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())