"""Tests for reusable inference and the thin Gradio adapter."""

from pathlib import Path

import pytest
import torch
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
	tensor = preprocess_image(Image.new("RGB", (80, 120), color=(255, 0, 0)))
	assert tuple(tensor.shape) == (1, 1, 150, 150)
	assert tensor.dtype == torch.float32
	assert float(tensor.min()) >= -1.0
	assert float(tensor.max()) <= 1.0


def test_inference_returns_probabilities_for_correct_classes():
	result = predict_image(Image.new("L", (150, 150), color=128))
	assert result["predicted_class"] in CLASS_NAMES
	assert 0.0 <= result["confidence"] <= 1.0
	assert tuple(result["probabilities"]) == CLASS_NAMES
	assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)


def test_gradio_adapter_preserves_uploaded_preview_and_label_scores():
	image = Image.new("L", (150, 150), color=128)
	preview, probabilities = predict_for_ui(image)
	assert preview is image
	assert tuple(probabilities) == CLASS_NAMES


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