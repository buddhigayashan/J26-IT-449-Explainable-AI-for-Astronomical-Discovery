"""Tests for deterministic input quality checks."""

import io

import numpy as np
import pytest
from PIL import Image

from src.ood.input_guard import prepare_guarded_image


def test_rgb_input_is_accepted_and_converted_to_classifier_shape():
	gradient = np.tile(np.arange(150, dtype=np.uint8), (150, 1))
	image = Image.fromarray(np.stack((gradient, gradient, gradient), axis=-1), mode="RGB")
	guarded = prepare_guarded_image(image)
	assert guarded.image.mode == "L"
	assert tuple(guarded.tensor.shape) == (1, 1, 150, 150)


@pytest.mark.parametrize("value", [0, 128, 255])
def test_blank_or_uniform_images_are_rejected(value):
	with pytest.raises(ValueError):
		prepare_guarded_image(Image.new("L", (150, 150), value))


def test_malformed_file_is_rejected():
	with pytest.raises(ValueError, match="valid image"):
		prepare_guarded_image(io.BytesIO(b"not an image"))