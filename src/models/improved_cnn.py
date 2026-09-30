"""Improved spatially preserving CNN for MiraBest binary classification."""

from __future__ import annotations

import torch
from torch import nn


class ImprovedCNN(nn.Module):
	"""Moderate CNN baseline that retains a 4x4 spatial representation."""

	def __init__(self, num_classes: int = 2, dropout: float = 0.35) -> None:
		super().__init__()
		self.features = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2),
			nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2),
			nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2),
		)
		self.spatial_pool = nn.AdaptiveAvgPool2d((4, 4))
		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Linear(128 * 4 * 4, 128),
			nn.ReLU(inplace=True),
			nn.Dropout(dropout),
			nn.Linear(128, num_classes),
		)

	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		"""Return two class logits for ``(N, 1, 150, 150)`` inputs."""

		return self.classifier(self.spatial_pool(self.features(inputs)))