"""Train the baseline CNN using the existing train/validation split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.data.download_mirabest import load_mirabest
from src.data.preprocess import PreprocessConfig
from src.data.split_dataset import build_preprocessed_splits
from src.models.train import TrainingConfig, train_model


def main() -> int:
	parser = argparse.ArgumentParser(description="Train the MiraBest baseline CNN.")
	parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
	args = parser.parse_args()

	config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
	dataset_root = PROJECT_ROOT / config["paths"]["mirabest_root"]
	split_path = PROJECT_ROOT / config["paths"]["splits"] / "split_indices.json"
	if not split_path.is_file():
		raise FileNotFoundError(
			f"Saved split indices not found: {split_path}. "
			"Run python -m src.data.split_dataset first."
		)

	dataset = load_mirabest(dataset_root, download=False)
	split_indices = json.loads(split_path.read_text(encoding="utf-8"))
	preprocessing_config = config["preprocessing"]
	preprocess_config = PreprocessConfig(
		normalize_mean=float(preprocessing_config["normalize_mean"]),
		normalize_std=float(preprocessing_config["normalize_std"]),
		rotation_degrees=float(preprocessing_config["rotation_degrees"]),
		horizontal_flip=bool(preprocessing_config["horizontal_flip"]),
		vertical_flip=bool(preprocessing_config["vertical_flip"]),
	)
	splits = build_preprocessed_splits(dataset, split_indices, preprocess_config)
	training_config = config["training"]
	result = train_model(
		splits["train"],
		splits["validation"],
		config=TrainingConfig(
			seed=int(training_config["seed"]),
			batch_size=int(training_config["batch_size"]),
			learning_rate=float(training_config["learning_rate"]),
			num_epochs=int(training_config["num_epochs"]),
			num_workers=int(training_config["num_workers"]),
			early_stopping_patience=int(training_config["early_stopping_patience"]),
			device=str(training_config["device"]),
		),
		output_root=PROJECT_ROOT / "outputs",
	)

	print("Baseline CNN training complete")
	print(f"Best epoch: {result['best_epoch']}")
	print(f"Best validation loss: {result['best_validation_loss']:.6f}")
	print(f"Best validation accuracy: {result['best_validation_accuracy']:.4f}")
	print(f"Checkpoint: {result['checkpoint_path']}")
	print("Test evaluation: not performed")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
