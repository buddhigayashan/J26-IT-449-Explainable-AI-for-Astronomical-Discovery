"""Tests for the baseline CNN and training utilities."""

import torch
from torch.utils.data import TensorDataset

from src.models.baseline_cnn import BaselineCNN
from src.models.train import TrainingConfig, train_model
from src.utils.seed import set_global_seed


def test_baseline_cnn_forward_shape():
	model = BaselineCNN()
	logits = model(torch.zeros(4, 1, 150, 150))
	assert logits.shape == (4, 2)
	assert torch.isfinite(logits).all()


def test_baseline_cnn_initialization_is_reproducible():
	set_global_seed(123)
	first_model = BaselineCNN()
	set_global_seed(123)
	second_model = BaselineCNN()
	assert all(
		torch.equal(first_parameter, second_parameter)
		for first_parameter, second_parameter in zip(first_model.parameters(), second_model.parameters())
	)


def test_training_saves_best_checkpoint_and_validation_artifacts(tmp_path):
	set_global_seed(42)
	train_images = torch.rand(4, 1, 150, 150)
	train_labels = torch.tensor([0, 1, 0, 1])
	validation_images = torch.rand(2, 1, 150, 150)
	validation_labels = torch.tensor([0, 1])
	result = train_model(
		TensorDataset(train_images, train_labels),
		TensorDataset(validation_images, validation_labels),
		config=TrainingConfig(
			seed=42,
			batch_size=2,
			learning_rate=0.001,
			num_epochs=1,
			early_stopping_patience=1,
			device="cpu",
		),
		output_root=tmp_path,
	)

	checkpoint = result["checkpoint_path"]
	assert checkpoint == tmp_path / "models" / "baseline_cnn_best.pth"
	assert checkpoint.is_file()
	checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
	assert "model_state_dict" in checkpoint_data
	assert (tmp_path / "metrics" / "training_history.csv").is_file()
	assert (tmp_path / "metrics" / "baseline_validation_metrics.json").is_file()
	assert (tmp_path / "figures" / "training_curves.png").is_file()
