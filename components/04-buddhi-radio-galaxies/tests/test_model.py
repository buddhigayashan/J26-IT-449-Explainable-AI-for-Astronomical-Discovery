"""Tests for the baseline CNN and training utilities."""

import torch
from torch.utils.data import TensorDataset

from src.models.baseline_cnn import BaselineCNN
from src.models.improved_cnn import ImprovedCNN
from src.models.resnet_cnn import ResNetCNN, ResidualBlock
from src.models.train import TrainingConfig, train_model, training_artifact_paths
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


def test_improved_cnn_forward_shape_and_parameter_size():
	model = ImprovedCNN()
	logits = model(torch.zeros(2, 1, 150, 150))
	parameter_count = sum(parameter.numel() for parameter in model.parameters())
	assert logits.shape == (2, 2)
	assert parameter_count > sum(parameter.numel() for parameter in BaselineCNN().parameters())
	assert parameter_count < 1_000_000


def test_improved_cnn_initialization_is_reproducible():
	set_global_seed(456)
	first_model = ImprovedCNN()
	set_global_seed(456)
	second_model = ImprovedCNN()
	assert all(
		torch.equal(first_parameter, second_parameter)
		for first_parameter, second_parameter in zip(first_model.parameters(), second_model.parameters())
	)


def test_resnet_cnn_forward_shape_and_parameter_size():
	model = ResNetCNN()
	logits = model(torch.zeros(2, 1, 150, 150))
	parameter_count = sum(parameter.numel() for parameter in model.parameters())
	assert logits.shape == (2, 2)
	assert parameter_count < 1_000_000
	assert isinstance(model.blocks[0], ResidualBlock)


def test_resnet_cnn_initialization_is_reproducible():
	set_global_seed(789)
	first_model = ResNetCNN()
	set_global_seed(789)
	second_model = ResNetCNN()
	assert all(
		torch.equal(first_parameter, second_parameter)
		for first_parameter, second_parameter in zip(first_model.parameters(), second_model.parameters())
	)


def test_improved_artifacts_are_separate_from_baseline():
	baseline_paths = training_artifact_paths("outputs", "baseline")
	improved_paths = training_artifact_paths("outputs", "improved")
	resnet_paths = training_artifact_paths("outputs", "resnet")
	assert improved_paths["checkpoint"].name == "improved_cnn_best.pth"
	assert improved_paths["history"].name == "improved_training_history.csv"
	assert improved_paths["validation_metrics"].name == "improved_validation_metrics.json"
	assert improved_paths["curves"].name == "improved_training_curves.png"
	assert resnet_paths["checkpoint"].name == "resnet_cnn_best.pth"
	assert resnet_paths["history"].name == "resnet_training_history.csv"
	assert resnet_paths["validation_metrics"].name == "resnet_validation_metrics.json"
	assert resnet_paths["curves"].name == "resnet_training_curves.png"
	assert set(baseline_paths.values()).isdisjoint(improved_paths.values())
	assert set(baseline_paths.values()).isdisjoint(resnet_paths.values())
	assert set(improved_paths.values()).isdisjoint(resnet_paths.values())


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
