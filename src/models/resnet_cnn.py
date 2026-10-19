"""Lightweight residual CNN for MiraBest binary classification."""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
	"""Two-convolution residual block with an optional projection shortcut."""

	def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
		super().__init__()
		self.convolution = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
		)
		if stride != 1 or in_channels != out_channels:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(out_channels),
			)
		else:
			self.shortcut = nn.Identity()
		self.activation = nn.ReLU(inplace=True)

	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		return self.activation(self.convolution(inputs) + self.shortcut(inputs))


class ResNetCNN(nn.Module):
	"""Small ResNet-style classifier returning logits for two classes."""

	def __init__(self, num_classes: int = 2, dropout: float = 0.30) -> None:
		super().__init__()
		self.stem = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2),
		)
		self.blocks = nn.Sequential(
			ResidualBlock(32, 32),
			ResidualBlock(32, 64, stride=2),
			ResidualBlock(64, 128, stride=2),
		)
		self.spatial_pool = nn.AdaptiveAvgPool2d((2, 2))
		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Linear(128 * 2 * 2, 64),
			nn.ReLU(inplace=True),
			nn.Dropout(dropout),
			nn.Linear(64, num_classes),
		)

	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		"""Return two class logits for ``(N, 1, 150, 150)`` inputs."""

		features = self.blocks(self.stem(inputs))
		return self.classifier(self.spatial_pool(features))