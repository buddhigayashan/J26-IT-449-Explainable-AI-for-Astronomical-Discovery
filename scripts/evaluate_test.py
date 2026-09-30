"""Evaluate the selected ResNet checkpoint on the test split exactly once."""

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
from src.data.preprocess import PreprocessConfig, build_transforms
from src.data.split_dataset import TransformedSubset
from src.models.evaluate import evaluate_model, get_test_indices, save_test_results
from src.models.resnet_cnn import ResNetCNN


def main() -> int:
	parser = argparse.ArgumentParser(
		description="Run the one-time final ResNet evaluation on the test split only."
	)
	parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
	args = parser.parse_args()

	config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
	dataset_root = PROJECT_ROOT / config["paths"]["mirabest_root"]
	split_path = PROJECT_ROOT / config["paths"]["splits"] / "split_indices.json"
	checkpoint_path = PROJECT_ROOT / "outputs" / "models" / "resnet_cnn_best.pth"
	if not split_path.is_file():
		raise FileNotFoundError(f"Saved split indices not found: {split_path}")
	if not checkpoint_path.is_file():
		raise FileNotFoundError(f"Selected checkpoint not found: {checkpoint_path}")

	dataset = load_mirabest(dataset_root, download=False)
	split_indices = json.loads(split_path.read_text(encoding="utf-8"))
	_, evaluation_transform = build_transforms(
		PreprocessConfig(
			normalize_mean=float(config["preprocessing"]["normalize_mean"]),
			normalize_std=float(config["preprocessing"]["normalize_std"]),
			rotation_degrees=0.0,
			horizontal_flip=False,
			vertical_flip=False,
		)
	)
	test_dataset = TransformedSubset(
		dataset,
		get_test_indices(split_indices),
		evaluation_transform,
	)
	metrics = evaluate_model(
		ResNetCNN(),
		checkpoint_path,
		test_dataset,
		batch_size=int(config["training"]["batch_size"]),
		device=str(config["training"]["device"]),
		split_name="test",
	)
	json_path, csv_path = save_test_results(metrics, checkpoint_path, PROJECT_ROOT / "outputs")
	print("Final ResNet test evaluation")
	print(f"Test samples evaluated: {metrics['num_test_samples']}")
	print(f"Test accuracy: {metrics['accuracy']:.6f}")
	print(f"Macro F1: {metrics['macro_f1']:.6f}")
	print(f"Balanced accuracy: {metrics['balanced_accuracy']:.6f}")
	print(f"Test loss: {metrics['test_loss']:.6f}")
	print(f"Evaluation JSON: {json_path}")
	print(f"Metrics CSV: {csv_path}")
	print("Training or fine-tuning performed: no")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())