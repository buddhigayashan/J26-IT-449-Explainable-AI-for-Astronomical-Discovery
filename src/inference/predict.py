"""Reusable CPU inference for the selected MiraBest ResNet classifier."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, TypedDict

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.models.resnet_cnn import ResNetCNN


CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "models" / "resnet_cnn_best.pth"
CLASS_NAMES = ("FRI", "FRII")


class PredictionResult(TypedDict):
	"""Structured output returned by :func:`predict_image`."""

	predicted_class: str
	confidence: float
	probabilities: dict[str, float]


def preprocess_image(image: Any) -> torch.Tensor:
	"""Convert an image to a normalized ``(1, 1, 150, 150)`` tensor."""

	if image is None:
		raise ValueError("Please upload one radio galaxy image.")
	try:
		pil_image = image if isinstance(image, Image.Image) else Image.open(image)
		pil_image = pil_image.convert("L").resize((150, 150), Image.Resampling.BILINEAR)
		tensor = transforms.ToTensor()(pil_image)
		return transforms.Normalize((0.5,), (0.5,))(tensor).unsqueeze(0)
	except (OSError, TypeError, UnidentifiedImageError, ValueError) as error:
		raise ValueError("The uploaded file is not a valid image.") from error


@lru_cache(maxsize=1)
def load_checkpoint(checkpoint_path: str = str(CHECKPOINT_PATH)) -> ResNetCNN:
	"""Load the selected checkpoint once and keep the model in evaluation mode."""

	path = Path(checkpoint_path)
	if not path.is_file():
		raise FileNotFoundError(f"Selected ResNet checkpoint was not found: {path}")
	model = ResNetCNN()
	checkpoint = torch.load(path, map_location="cpu", weights_only=False)
	state_dict = checkpoint.get("model_state_dict", checkpoint)
	model.load_state_dict(state_dict)
	model.eval()
	return model


def predict_image(image: Any) -> PredictionResult:
	"""Predict FR-I/FR-II probabilities for one image using the saved checkpoint."""

	tensor = preprocess_image(image)
	model = load_checkpoint()
	with torch.no_grad():
		probabilities_tensor = torch.softmax(model(tensor), dim=1)[0]
	probabilities = {
		CLASS_NAMES[index]: float(probabilities_tensor[index])
		for index in range(len(CLASS_NAMES))
	}
	predicted_class = max(probabilities, key=probabilities.get)
	return {
		"predicted_class": predicted_class,
		"confidence": probabilities[predicted_class],
		"probabilities": probabilities,
	}