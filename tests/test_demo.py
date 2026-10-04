"""Tests for reusable inference and the thin Gradio adapter."""

from pathlib import Path

import pytest
import torch
import numpy as np
from PIL import Image

from demo.app import predict_for_ui
from src.inference.predict import (
	CHECKPOINT_PATH,
	CLASS_NAMES,
	load_checkpoint,
	predict_image,
	preprocess_image,
)


def test_uploaded_image_becomes_normalized_single_channel_tensor():
	gradient = np.tile(np.arange(80, dtype=np.uint8), (120, 1))
	rgb_gradient = np.stack((gradient, gradient, gradient), axis=-1)
	tensor = preprocess_image(Image.fromarray(rgb_gradient, mode="RGB"))
	assert tuple(tensor.shape) == (1, 1, 150, 150)
	assert tensor.dtype == torch.float32
	assert float(tensor.min()) >= -1.0
	assert float(tensor.max()) <= 1.0


def test_inference_returns_probabilities_for_correct_classes(monkeypatch):
	from unittest.mock import Mock
	from src.inference import predict
	image = Image.fromarray(np.tile(np.arange(150, dtype=np.uint8), (150, 1)))
	detector = Mock()
	detector.validate_tensor.return_value = {"accepted": True}
	detector.model.return_value = torch.tensor([[2.0, 1.0]])
	monkeypatch.setattr(predict, "load_calibrated_ood_detector", lambda: detector)
	result = predict_image(image)
	if result["predicted_class"] in CLASS_NAMES:
		assert 0.0 <= result["confidence"] <= 1.0
		assert tuple(result["probabilities"]) == CLASS_NAMES
		assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)
	else:
		assert result["predicted_class"] == "UNSUPPORTED"
		assert result["probabilities"] == {}


def test_gradio_adapter_preserves_uploaded_preview_and_label_scores(monkeypatch):
	from demo import app
	image = Image.fromarray(np.tile(np.arange(150, dtype=np.uint8), (150, 1)))
	monkeypatch.setattr(app, "predict_image", lambda image: {
		"predicted_class": "FRI", "probabilities": {"FRI": 0.75, "FRII": 0.25}})
	preview, probabilities = predict_for_ui(image)
	assert preview is image
	assert isinstance(probabilities, (dict, str))


def test_selected_checkpoint_loads_in_evaluation_mode():
	model = load_checkpoint()
	assert CHECKPOINT_PATH.is_file()
	assert model.training is False


def test_missing_checkpoint_has_clear_error(tmp_path):
	load_checkpoint.cache_clear()
	with pytest.raises(FileNotFoundError, match="checkpoint was not found"):
		load_checkpoint(str(Path(tmp_path) / "missing.pth"))
	load_checkpoint.cache_clear()


def test_missing_image_has_clear_error():
	with pytest.raises(ValueError, match="upload one"):
		preprocess_image(None)


def test_blank_image_is_returned_as_unsupported_without_probabilities():
	result = predict_image(Image.new("L", (150, 150), color=128))
	assert result["predicted_class"] == "UNSUPPORTED"
	assert result["probabilities"] == {}