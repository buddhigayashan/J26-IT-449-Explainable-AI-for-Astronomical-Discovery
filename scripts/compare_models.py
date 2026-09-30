"""Compare baseline, improved, and ResNet models on validation data only."""

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
from src.data.preprocess import build_transforms
from src.data.split_dataset import TransformedSubset
from src.models.baseline_cnn import BaselineCNN
from src.models.evaluate import evaluate_model, get_validation_indices, save_comparison_results
from src.models.improved_cnn import ImprovedCNN
from src.models.resnet_cnn import ResNetCNN


def main() -> int:
	parser = argparse.ArgumentParser(
		description="Evaluate all trained CNN variants on the validation split only."
	)
	parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
	args = parser.parse_args()

	config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
	dataset_root = PROJECT_ROOT / config["paths"]["mirabest_root"]
	split_path = PROJECT_ROOT / config["paths"]["splits"] / "split_indices.json"
	if not split_path.is_file():
		raise FileNotFoundError(f"Saved split indices not found: {split_path}")

	dataset = load_mirabest(dataset_root, download=False)
	split_indices = json.loads(split_path.read_text(encoding="utf-8"))
	_, evaluation_transform = build_transforms()
	validation_dataset = TransformedSubset(
		dataset,
		get_validation_indices(split_indices),
		evaluation_transform,
	)
	batch_size = int(config["training"]["batch_size"])
	device = str(config["training"]["device"])
	model_specs = {
		"baseline": (BaselineCNN(), PROJECT_ROOT / "outputs" / "models" / "baseline_cnn_best.pth"),
		"improved": (ImprovedCNN(), PROJECT_ROOT / "outputs" / "models" / "improved_cnn_best.pth"),
		"resnet": (ResNetCNN(), PROJECT_ROOT / "outputs" / "models" / "resnet_cnn_best.pth"),
	}
	results = {}
	for model_name, (model, checkpoint_path) in model_specs.items():
		if not checkpoint_path.is_file():
			raise FileNotFoundError(f"Checkpoint not found for {model_name}: {checkpoint_path}")
		results[model_name] = evaluate_model(
			model,
			checkpoint_path,
			validation_dataset,
			batch_size=batch_size,
			device=device,
		)

	save_comparison_results(results, PROJECT_ROOT / "outputs")
	print("Validation-only model comparison")
	print(f"Validation samples evaluated per model: {len(validation_dataset)}")
	print("Test split evaluated: no")
	for model_name, metrics in results.items():
		print(
			f"{model_name.title()}: accuracy={metrics['accuracy']:.4f}, "
			f"macro_f1={metrics['macro_f1']:.4f}, "
			f"balanced_accuracy={metrics['balanced_accuracy']:.4f}, "
			f"validation_loss={metrics['validation_loss']:.6f}"
		)
	print(f"Strongest Macro F1: {max(results, key=lambda name: results[name]['macro_f1'])}")
	print(
		"Strongest Balanced Accuracy: "
		f"{max(results, key=lambda name: results[name]['balanced_accuracy'])}"
	)
	print(f"Lowest Validation Loss: {min(results, key=lambda name: results[name]['validation_loss'])}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())