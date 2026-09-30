"""Tests for validation-only model comparison utilities."""

import torch
from torch import nn
from torch.utils.data import TensorDataset

from src.models.evaluate import (
	METRIC_KEYS,
	TEST_METRIC_KEYS,
	evaluate_model,
	get_test_indices,
	get_validation_indices,
)


class TinyClassifier(nn.Module):
	def __init__(self):
		super().__init__()
		self.classifier = nn.Linear(4, 2)

	def forward(self, inputs):
		return self.classifier(inputs.flatten(start_dim=1))


def test_validation_metrics_have_required_keys_and_confusion_shape(tmp_path):
	model = TinyClassifier()
	checkpoint_path = tmp_path / "tiny.pth"
	torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
	dataset = TensorDataset(
		torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0]]]]),
		torch.tensor([0, 1]),
	)
	metrics = evaluate_model(model, checkpoint_path, dataset, batch_size=2)
	assert METRIC_KEYS <= metrics.keys()
	assert len(metrics["confusion_matrix"]) == 2
	assert all(len(row) == 2 for row in metrics["confusion_matrix"])


def test_validation_indices_are_shared_without_accessing_test_indices():
	class TestSplitSentinel:
		def __iter__(self):
			raise AssertionError("The test split must not be accessed.")

	split_indices = {"validation": [4, 8, 12], "test": TestSplitSentinel()}
	first_view = get_validation_indices(split_indices)
	second_view = get_validation_indices(split_indices)
	assert first_view == second_view == [4, 8, 12]


def test_test_evaluation_uses_only_test_indices():
	class ValidationSplitSentinel:
		def __iter__(self):
			raise AssertionError("The validation split must not be accessed.")

	split_indices = {"test": [2, 5, 9], "validation": ValidationSplitSentinel()}
	assert get_test_indices(split_indices) == [2, 5, 9]


def test_test_metrics_use_saved_checkpoint_without_optimizer(tmp_path, monkeypatch):
	model = TinyClassifier()
	checkpoint_path = tmp_path / "resnet.pth"
	torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
	dataset = TensorDataset(
		torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0]]]]),
		torch.tensor([0, 1]),
	)

	def fail_if_optimizer_created(*args, **kwargs):
		raise AssertionError("Test evaluation must not create an optimizer.")

	monkeypatch.setattr(torch.optim, "Adam", fail_if_optimizer_created)
	metrics = evaluate_model(
		model,
		checkpoint_path,
		dataset,
		batch_size=2,
		split_name="test",
	)
	assert TEST_METRIC_KEYS <= metrics.keys()
	assert model.training is False
	assert len(metrics["confusion_matrix"]) == 2
	assert all(len(row) == 2 for row in metrics["confusion_matrix"])
	assert not (METRIC_KEYS - {"num_validation_samples", "validation_loss"}) - metrics.keys()