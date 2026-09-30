"""Compact baseline CNN for MiraBest binary classification."""

from __future__ import annotations

import torch
from torch import nn


class BaselineCNN(nn.Module):
	"""Small convolutional classifier returning logits for two classes."""

	def __init__(self, num_classes: int = 2, dropout: float = 0.30) -> None:
		super().__init__()
		self.features = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2),
			nn.Conv2d(16, 32, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2),
			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2),
		)
		self.pool = nn.AdaptiveAvgPool2d((1, 1))
		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Dropout(dropout),
			nn.Linear(64, num_classes),
		)

	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		"""Return unnormalized class logits for ``(N, 1, 150, 150)`` inputs."""

		return self.classifier(self.pool(self.features(inputs)))
