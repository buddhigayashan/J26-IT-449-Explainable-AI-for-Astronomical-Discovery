"""Preprocessing transforms for MiraBest grayscale radio images."""

from __future__ import annotations

from dataclasses import dataclass

from torchvision import transforms


@dataclass(frozen=True)
class PreprocessConfig:
	"""Configuration for deterministic evaluation and train transforms."""

	normalize_mean: float = 0.5
	normalize_std: float = 0.5
	rotation_degrees: float = 15.0
	horizontal_flip: bool = True
	vertical_flip: bool = True


def _base_transforms(config: PreprocessConfig) -> list:
	return [
		transforms.ToTensor(),
		transforms.Normalize((config.normalize_mean,), (config.normalize_std,)),
	]


def build_transforms(
	config: PreprocessConfig | None = None,
) -> tuple[transforms.Compose, transforms.Compose]:
	"""Return ``(train_transform, evaluation_transform)``.

	Images remain 150x150. Only the training transform includes random
	morphology-preserving operations; validation and test use deterministic
	conversion and normalization only.
	"""

	config = config or PreprocessConfig()
	train_operations = []
	if config.rotation_degrees > 0:
		train_operations.append(transforms.RandomRotation(config.rotation_degrees))
	if config.horizontal_flip:
		train_operations.append(transforms.RandomHorizontalFlip())
	if config.vertical_flip:
		train_operations.append(transforms.RandomVerticalFlip())

	return (
		transforms.Compose(train_operations + _base_transforms(config)),
		transforms.Compose(_base_transforms(config)),
	)
