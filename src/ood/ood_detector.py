"""Transparent energy and feature-distance OOD detector for the locked ResNet."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class OODThresholds:
	"""Validation-calibrated upper thresholds for both OOD scores."""

	energy_threshold: float
	feature_distance_threshold: float
	validation_percentile: float


class OODDetector:
	"""Energy plus diagonal Mahalanobis-style feature-distance detector."""

	def __init__(self, model: nn.Module, batch_size: int = 32) -> None:
		self.model = model
		self.batch_size = batch_size
		self.feature_mean: torch.Tensor | None = None
		self.feature_variance: torch.Tensor | None = None
		self.thresholds: OODThresholds | None = None

	def _features_and_logits(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		if not hasattr(self.model, "stem") or not hasattr(self.model, "blocks"):
			raise TypeError("OODDetector requires the project ResNet-style model.")
		features = self.model.blocks(self.model.stem(images))
		pooled = self.model.spatial_pool(features)
		return pooled.flatten(start_dim=1), self.model.classifier(pooled)

	def _collect_features(self, dataset: Dataset) -> tuple[torch.Tensor, torch.Tensor]:
		loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
		features: list[torch.Tensor] = []
		energies: list[torch.Tensor] = []
		self.model.eval()
		with torch.no_grad():
			for images, _ in loader:
				feature_batch, logits = self._features_and_logits(images)
				features.append(feature_batch.cpu())
				energies.append((-torch.logsumexp(logits, dim=1)).cpu())
		if not features:
			raise ValueError("Cannot fit OOD detector on an empty dataset.")
		return torch.cat(features), torch.cat(energies)

	def fit_feature_statistics(self, training_dataset: Dataset) -> None:
		"""Fit feature mean/variance using training split features only."""

		features, _ = self._collect_features(training_dataset)
		self.feature_mean = features.mean(dim=0)
		self.feature_variance = features.var(dim=0, unbiased=False).clamp_min(1e-6)

	def score_dataset(self, dataset: Dataset) -> tuple[torch.Tensor, torch.Tensor]:
		"""Score a deterministic dataset as energy and feature distance arrays."""

		if self.feature_mean is None or self.feature_variance is None:
			raise RuntimeError("Fit feature statistics before scoring.")
		features, energies = self._collect_features(dataset)
		distance = torch.sqrt(
			(((features - self.feature_mean) ** 2) / self.feature_variance).sum(dim=1)
		)
		return energies, distance

	def calibrate(self, validation_dataset: Dataset, percentile: float = 0.99) -> OODThresholds:
		"""Choose upper thresholds from validation in-distribution scores only."""

		if not 0.5 < percentile < 1.0:
			raise ValueError("Calibration percentile must be between 0.5 and 1.0.")
		energies, distances = self.score_dataset(validation_dataset)
		self.thresholds = OODThresholds(
			energy_threshold=float(torch.quantile(energies, percentile)),
			feature_distance_threshold=float(torch.quantile(distances, percentile)),
			validation_percentile=percentile,
		)
		return self.thresholds

	def score_tensor(self, tensor: torch.Tensor) -> dict[str, float]:
		if self.feature_mean is None or self.feature_variance is None:
			raise RuntimeError("Fit feature statistics before scoring.")
		self.model.eval()
		with torch.no_grad():
			features, logits = self._features_and_logits(tensor)
			energy = float((-torch.logsumexp(logits, dim=1))[0].item())
			distance = float(
				torch.sqrt(
					(((features[0].cpu() - self.feature_mean) ** 2) / self.feature_variance).sum()
				).item()
			)
		return {"energy_score": energy, "feature_distance": distance}

	def validate_tensor(self, tensor: torch.Tensor) -> dict[str, Any]:
		"""Return structured acceptance information for one guarded tensor."""

		if self.thresholds is None:
			raise RuntimeError("Calibrate OOD thresholds before validating inputs.")
		scores = self.score_tensor(tensor)
		if not all(math.isfinite(value) for value in scores.values()):
			raise ValueError("Non-finite OOD scores.")
		energy_rejected = scores["energy_score"] > self.thresholds.energy_threshold
		distance_rejected = scores["feature_distance"] > self.thresholds.feature_distance_threshold
		reasons = []
		if energy_rejected:
			reasons.append("energy score exceeds validation threshold")
		if distance_rejected:
			reasons.append("feature distance exceeds validation threshold")
		return {
			"accepted": not reasons,
			"reason": "accepted" if not reasons else "; ".join(reasons),
			**scores,
			"ood_status": "in_distribution" if not reasons else "out_of_distribution",
			"thresholds": asdict(self.thresholds),
		}