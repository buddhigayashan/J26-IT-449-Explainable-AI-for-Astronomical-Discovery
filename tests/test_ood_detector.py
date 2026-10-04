"""Tests for energy and feature-distance OOD detection."""

import hashlib
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

from src.inference.predict import CLASS_NAMES
from src.models.resnet_cnn import ResNetCNN
from src.ood.ood_detector import OODDetector


def test_detector_fits_training_features_and_calibrates_validation_thresholds():
	model = ResNetCNN()
	detector = OODDetector(model, batch_size=2)
	training = TensorDataset(torch.rand(4, 1, 150, 150), torch.tensor([0, 1, 0, 1]))
	validation = TensorDataset(torch.rand(4, 1, 150, 150), torch.tensor([0, 1, 0, 1]))
	detector.fit_feature_statistics(training)
	thresholds = detector.calibrate(validation)
	result = detector.validate_tensor(torch.rand(1, 1, 150, 150))
	assert thresholds.validation_percentile == 0.99
	assert "energy_score" in result
	assert "feature_distance" in result
	assert result["ood_status"] in {"in_distribution", "out_of_distribution"}


def test_detector_requires_calibration_before_validation():
	detector = OODDetector(ResNetCNN())
	try:
		detector.validate_tensor(torch.rand(1, 1, 150, 150))
	except RuntimeError as error:
		assert "Calibrate" in str(error)
	else:
		raise AssertionError("Uncalibrated OOD detector unexpectedly accepted validation input.")


def test_ood_outputs_are_isolated_from_final_test_artifacts():
	from scripts.run_ood_evaluation import _paths

	paths = _paths("outputs")
	assert paths["json"].as_posix().endswith("outputs/ood/ood_evaluation.json")
	assert paths["csv"].as_posix().endswith("outputs/ood/ood_predictions.csv")
	assert paths["json"] != "outputs/metrics/resnet_test_evaluation.json"
	assert paths["csv"] != "outputs/metrics/final_test_metrics.csv"


def test_locked_checkpoint_and_final_metrics_exist_and_are_not_ood_outputs():
	checkpoint = Path("outputs/models/resnet_cnn_best.pth")
	final_metrics = Path("outputs/metrics/resnet_test_evaluation.json")
	final_csv = Path("outputs/metrics/final_test_metrics.csv")
	assert checkpoint.is_file()
	assert final_metrics.is_file()
	assert final_csv.is_file()
	assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()


def test_application_has_exactly_three_final_states():
	assert set(CLASS_NAMES) | {"UNSUPPORTED"} == {"FRI", "FRII", "UNSUPPORTED"}


def test_detector_uses_no_optimizer_or_training_mode():
	detector = OODDetector(ResNetCNN())
	assert not hasattr(detector, "optimizer")
	assert not hasattr(detector, "train")