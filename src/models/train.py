"""Baseline CNN training loop and validation-only reporting."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.models.baseline_cnn import BaselineCNN
from src.utils.seed import set_global_seed


@dataclass(frozen=True)
class TrainingConfig:
	"""Configurable training settings shared by the model variants."""

	seed: int = 42
	batch_size: int = 32
	learning_rate: float = 0.001
	weight_decay: float = 0.0
	num_epochs: int = 20
	num_workers: int = 0
	early_stopping_patience: int = 5
	scheduler_patience: int | None = None
	scheduler_factor: float = 0.5
	device: str = "cpu"


def _run_epoch(
	model: nn.Module,
	loader: DataLoader,
	criterion: nn.Module,
	device: torch.device,
	optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
	training = optimizer is not None
	model.train(training)
	total_loss = 0.0
	correct = 0
	total = 0
	for images, labels in loader:
		images = images.to(device)
		labels = labels.to(device)
		if training:
			optimizer.zero_grad(set_to_none=True)
		logits = model(images)
		loss = criterion(logits, labels)
		if training:
			loss.backward()
			optimizer.step()
		total_loss += loss.item() * labels.size(0)
		correct += int((logits.argmax(dim=1) == labels).sum().item())
		total += labels.size(0)

	if total == 0:
		raise ValueError("Cannot train or validate with an empty dataset.")
	return total_loss / total, correct / total


def _save_history(history: list[dict[str, float | int]], path: Path) -> None:
	with path.open("w", newline="", encoding="utf-8") as file_handle:
		writer = csv.DictWriter(file_handle, fieldnames=list(history[0]))
		writer.writeheader()
		writer.writerows(history)


def _save_curves(history: list[dict[str, float | int]], path: Path) -> None:
	epochs = [row["epoch"] for row in history]
	figure, axes = plt.subplots(1, 2, figsize=(11, 4))
	axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train")
	axes[0].plot(epochs, [row["validation_loss"] for row in history], label="Validation")
	axes[0].set_title("Loss")
	axes[0].set_xlabel("Epoch")
	axes[0].legend()
	axes[1].plot(epochs, [row["train_accuracy"] for row in history], label="Train")
	axes[1].plot(epochs, [row["validation_accuracy"] for row in history], label="Validation")
	axes[1].set_title("Accuracy")
	axes[1].set_xlabel("Epoch")
	axes[1].legend()
	figure.tight_layout()
	figure.savefig(path, dpi=150)
	plt.close(figure)


def training_artifact_paths(
	output_root: str | Path, artifact_prefix: str = "baseline"
) -> dict[str, Path]:
	"""Return model-specific output paths without overwriting other runs."""

	output_root = Path(output_root)
	if artifact_prefix == "baseline":
		history_name = "training_history.csv"
		curves_name = "training_curves.png"
	else:
		history_name = f"{artifact_prefix}_training_history.csv"
		curves_name = f"{artifact_prefix}_training_curves.png"
	return {
		"checkpoint": output_root / "models" / f"{artifact_prefix}_cnn_best.pth",
		"history": output_root / "metrics" / history_name,
		"curves": output_root / "figures" / curves_name,
		"validation_metrics": output_root / "metrics" / f"{artifact_prefix}_validation_metrics.json",
	}


def train_model(
	train_dataset: Dataset,
	validation_dataset: Dataset,
	config: TrainingConfig | None = None,
	model: nn.Module | None = None,
	output_root: str | Path = "outputs",
	artifact_prefix: str = "baseline",
) -> dict[str, Any]:
	"""Train a selected model using train/validation data only.

	The test dataset is intentionally not accepted by this API. Only the best
	validation-loss checkpoint is persisted. ``artifact_prefix`` isolates model
	outputs, preserving baseline results when training an improved model.
	"""

	config = config or TrainingConfig()
	set_global_seed(config.seed)
	device = torch.device(config.device)
	if device.type == "cuda" and not torch.cuda.is_available():
		raise RuntimeError("CUDA was requested but is unavailable.")

	train_generator = torch.Generator()
	train_generator.manual_seed(config.seed)
	train_loader = DataLoader(
		train_dataset,
		batch_size=config.batch_size,
		shuffle=True,
		num_workers=config.num_workers,
		generator=train_generator,
	)
	validation_loader = DataLoader(
		validation_dataset,
		batch_size=config.batch_size,
		shuffle=False,
		num_workers=config.num_workers,
	)

	model = BaselineCNN() if model is None else model
	model.to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(
		model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
	)
	scheduler = (
		torch.optim.lr_scheduler.ReduceLROnPlateau(
			optimizer,
			mode="min",
			factor=config.scheduler_factor,
			patience=config.scheduler_patience,
		)
		if config.scheduler_patience is not None
		else None
	)
	output_root = Path(output_root)
	model_dir = output_root / "models"
	metrics_dir = output_root / "metrics"
	figures_dir = output_root / "figures"
	for directory in (model_dir, metrics_dir, figures_dir):
		directory.mkdir(parents=True, exist_ok=True)
	paths = training_artifact_paths(output_root, artifact_prefix)
	checkpoint_path = paths["checkpoint"]
	history_path = paths["history"]
	curves_path = paths["curves"]
	validation_metrics_path = paths["validation_metrics"]

	history: list[dict[str, float | int]] = []
	best_validation_loss = float("inf")
	best_epoch = 0
	patience_counter = 0
	for epoch in range(1, config.num_epochs + 1):
		train_loss, train_accuracy = _run_epoch(
			model, train_loader, criterion, device, optimizer
		)
		validation_loss, validation_accuracy = _run_epoch(
			model, validation_loader, criterion, device
		)
		if scheduler is not None:
			scheduler.step(validation_loss)
		epoch_metrics = {
			"epoch": epoch,
			"train_loss": train_loss,
			"validation_loss": validation_loss,
			"train_accuracy": train_accuracy,
			"validation_accuracy": validation_accuracy,
			"learning_rate": optimizer.param_groups[0]["lr"],
		}
		history.append(epoch_metrics)
		if validation_loss < best_validation_loss:
			best_validation_loss = validation_loss
			best_epoch = epoch
			patience_counter = 0
			torch.save(
				{
					"model_state_dict": model.state_dict(),
					"epoch": epoch,
					"validation_loss": validation_loss,
					"validation_accuracy": validation_accuracy,
					"config": asdict(config),
				},
				checkpoint_path,
			)
		else:
			patience_counter += 1
			if patience_counter >= config.early_stopping_patience:
				break

	_save_history(history, history_path)
	_save_curves(history, curves_path)
	validation_metrics = {
		"dataset": "MiraBest Zenodo v1 basic binary dataset",
		"model": artifact_prefix,
		"best_epoch": best_epoch,
		"best_validation_loss": best_validation_loss,
		"best_validation_accuracy": history[best_epoch - 1]["validation_accuracy"],
		"epochs_completed": len(history),
		"test_evaluation_performed": False,
		"history": history,
	}
	validation_metrics_path.write_text(
		json.dumps(validation_metrics, indent=2) + "\n", encoding="utf-8"
	)
	return {
		"history": history,
		"best_epoch": best_epoch,
		"best_validation_loss": best_validation_loss,
		"best_validation_accuracy": validation_metrics["best_validation_accuracy"],
		"checkpoint_path": checkpoint_path,
		"history_path": history_path,
		"curves_path": curves_path,
		"validation_metrics_path": validation_metrics_path,
	}
