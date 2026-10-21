"""Deterministic image-domain and input-quality checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
from torchvision import transforms


@dataclass(frozen=True)
class GuardedImage:
	"""Validated preview and normalized classifier input."""

	image: Image.Image
	tensor: torch.Tensor


def prepare_guarded_image(image: Any) -> GuardedImage:
	"""Open, validate, grayscale, resize, and normalize one input image."""

	if image is None:
		raise ValueError("Please upload one radio galaxy image.")
	try:
		pil_image = image if isinstance(image, Image.Image) else Image.open(image)
		if pil_image.width <= 0 or pil_image.height <= 0:
			raise ValueError("Image dimensions must be greater than zero.")
		grayscale = pil_image.convert("L")
		pixels = np.asarray(grayscale, dtype=np.float32) / 255.0
		if not np.isfinite(pixels).all():
			raise ValueError("Image contains non-finite pixel values.")
		if float(pixels.max()) == 0.0 or float(pixels.min()) == 1.0:
			raise ValueError("Completely blank images are not supported.")
		if float(pixels.var()) < 1e-6:
			raise ValueError("Near-uniform images are outside the supported input domain.")
		preview = grayscale.resize((150, 150), Image.Resampling.BILINEAR)
		tensor = transforms.ToTensor()(preview)
		tensor = transforms.Normalize((0.5,), (0.5,))(tensor).unsqueeze(0)
		return GuardedImage(preview, tensor)
	except (OSError, TypeError, UnidentifiedImageError, ValueError) as error:
		if isinstance(error, ValueError):
			raise
		raise ValueError("The uploaded file is not a valid image.") from error