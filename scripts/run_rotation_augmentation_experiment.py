"""Compare two ResNet rotation strategies using train/validation only."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")  # Save experiment figures without requiring a GUI/Tk installation.
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as TF
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.download_mirabest import load_mirabest
from src.data.split_dataset import TransformedSubset
from src.models.evaluate import CLASS_NAMES, evaluate_model
from src.models.resnet_cnn import ResNetCNN
from src.models.train import TrainingConfig, train_model
from src.ood.calibration import validate_splits
from src.utils.seed import set_global_seed

SEED = 42
STRATEGIES = ("current_augmentation", "right_angle_rotation")
RESULT_FIELDS = (
    "configuration", "seed", "best_epoch", "validation_loss", "validation_accuracy",
    "macro_precision", "macro_recall", "macro_f1", "balanced_accuracy",
)


class RandomRightAngleRotation:
    """Uniform choice of one exact right-angle rotation, using the seeded torch RNG."""

    angles = (0, 90, 180, 270)

    def __call__(self, image):
        angle = self.angles[int(torch.randint(4, (1,)).item())]
        return TF.rotate(image, angle)


def build_experiment_transforms(strategy, mean=0.5, std=0.5):
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown rotation strategy: {strategy}")
    rotation = (transforms.RandomRotation(15) if strategy == "current_augmentation"
                else RandomRightAngleRotation())
    # Match the existing classifier transforms; MiraBest is already grayscale 150x150.
    base = [transforms.ToTensor(), transforms.Normalize((mean,), (std,))]
    return (transforms.Compose([rotation, transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip()] + base),
            transforms.Compose(base))


def training_settings(config):
    """Use the selected ResNet's existing optimizer/scheduler/early-stopping settings."""
    base = config["training"]
    resnet = config.get("models", {}).get("resnet", {})
    return TrainingConfig(
        seed=SEED, batch_size=int(base["batch_size"]),
        learning_rate=float(resnet.get("learning_rate", base["learning_rate"])),
        weight_decay=float(resnet.get("weight_decay", base.get("weight_decay", 0.0))),
        num_epochs=int(resnet.get("num_epochs", base["num_epochs"])),
        num_workers=int(base["num_workers"]),
        early_stopping_patience=int(resnet.get("early_stopping_patience", base["early_stopping_patience"])),
        scheduler_patience=int(resnet["scheduler_patience"]) if "scheduler_patience" in resnet else None,
        scheduler_factor=float(resnet.get("scheduler_factor", 0.5)),
        device=str(base["device"]),
    )


def experiment_paths(output_root, strategy):
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown rotation strategy: {strategy}")
    root = Path(output_root)
    results = root / "augmentation_experiment"
    return {
        "checkpoint": root / "models" / "augmentation_experiment" / f"{strategy}_best.pth",
        "history": results / f"{strategy}_training_history.csv",
        "curves": results / f"{strategy}_training_curves.png",
        "validation_metrics": results / f"{strategy}_validation_metrics.json",
    }


def run_configuration(strategy, dataset, splits, config, settings, output_root):
    train_transform, validation_transform = build_experiment_transforms(
        strategy, float(config["preprocessing"]["normalize_mean"]),
        float(config["preprocessing"]["normalize_std"]))
    train_view = TransformedSubset(dataset, splits["train"], train_transform)
    validation_view = TransformedSubset(dataset, splits["validation"], validation_transform)
    paths = experiment_paths(output_root, strategy)
    # Reset before initialization for identical initial weights and loader seed in both runs.
    set_global_seed(SEED)
    training = train_model(train_view, validation_view, config=settings, model=ResNetCNN(),
                           output_root=output_root, artifact_prefix=strategy, artifact_paths=paths)
    metrics = evaluate_model(ResNetCNN(), paths["checkpoint"], validation_view,
                             batch_size=settings.batch_size, device=settings.device,
                             split_name="validation")
    return {
        "configuration": strategy, "seed": SEED, "best_epoch": training["best_epoch"],
        "validation_loss": metrics["validation_loss"], "validation_accuracy": metrics["accuracy"],
        **{name: metrics[name] for name in ("macro_precision", "macro_recall", "macro_f1", "balanced_accuracy")},
    }


def run_experiment(config_path, output_root=PROJECT_ROOT / "outputs"):
    output_root = Path(output_root)
    results_dir = output_root / "augmentation_experiment"
    csv_path = results_dir / "rotation_results.csv"
    metadata_path = results_dir / "experiment_config.json"
    figure_path = output_root / "figures" / "rotation_augmentation_comparison.png"
    destinations = [csv_path, metadata_path, figure_path]
    for strategy in STRATEGIES:
        destinations.extend(experiment_paths(output_root, strategy).values())
    # Refuse a repeated run instead of overwriting any existing experiment artifacts.
    for path in destinations:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing experiment artifact: {path}")
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    split_path = PROJECT_ROOT / config["paths"]["splits"] / "split_indices.json"
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    validate_splits(splits)  # Test indices are inspected for disjointness only.
    settings = training_settings(config)
    dataset = load_mirabest(PROJECT_ROOT / config["paths"]["mirabest_root"], download=False)
    results_dir.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps({
        "architecture": "ResNetCNN", "training": asdict(settings), "optimizer": "Adam",
        "preprocessing": {"normalize_mean": config["preprocessing"]["normalize_mean"],
                          "normalize_std": config["preprocessing"]["normalize_std"],
                          "horizontal_flip": True, "vertical_flip": True},
        "rotations": {STRATEGIES[0]: [-15, 15], STRATEGIES[1]: [0, 90, 180, 270]},
        "class_mapping": dict(enumerate(CLASS_NAMES)),
        "split_reference": str(split_path.resolve()),
        "train_indices": splits["train"], "validation_indices": splits["validation"],
        "test_evaluation_performed": False,
        "checkpoint_selection": "minimum validation loss",
    }, indent=2) + "\n", encoding="utf-8")
    results = [run_configuration(s, dataset, splits, config, settings, output_root) for s in STRATEGIES]
    with csv_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(results)
    figure, axis = plt.subplots(figsize=(9, 5))
    positions = np.arange(2)
    width = 0.25
    for offset, metric in enumerate(("validation_accuracy", "macro_f1", "balanced_accuracy")):
        axis.bar(positions + (offset - 1) * width, [r[metric] for r in results], width,
                 label=metric.replace("_", " ").title())
    axis.set_xticks(positions, STRATEGIES)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Validation score")
    axis.set_title("ResNet rotation augmentation: validation only, seed 42")
    axis.legend()
    figure.tight_layout()
    figure.savefig(figure_path, dpi=150)
    plt.close(figure)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    args = parser.parse_args()
    print(json.dumps(run_experiment(args.config), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
