"""Reusable CPU inference for the selected MiraBest ResNet classifier."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, TypedDict

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.models.resnet_cnn import ResNetCNN
from src.ood.input_guard import prepare_guarded_image
from src.ood.ood_detector import OODDetector
from src.ood.calibration import load_calibration, checkpoint_sha256


CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "models" / "resnet_cnn_best.pth"
CALIBRATION_PATH = PROJECT_ROOT / "outputs" / "ood" / "ood_calibration.pt"
CLASS_NAMES = ("FRI", "FRII")


class PredictionResult(TypedDict):
	"""Structured output returned by :func:`predict_image`."""

	predicted_class: str
	confidence: float | None
	probabilities: dict[str, float]
	message: str
	ood: dict[str, Any] | None


def preprocess_image(image: Any) -> torch.Tensor:
	"""Convert an image to a normalized ``(1, 1, 150, 150)`` tensor."""

	return prepare_guarded_image(image).tensor


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


def load_calibrated_ood_detector() -> OODDetector:
	"""Load frozen calibration; never fit statistics or thresholds at runtime."""
	# Validate provenance before loading the model, including on repeated requests.
	artifact = load_calibration(CALIBRATION_PATH, CHECKPOINT_PATH)
	load_checkpoint.cache_clear()
	detector = OODDetector(load_checkpoint(str(CHECKPOINT_PATH)))
	if checkpoint_sha256(CHECKPOINT_PATH) != artifact["checkpoint_sha256"]:
		raise ValueError("Checkpoint hash mismatch during model loading.")
	detector.feature_mean = artifact["feature_mean"]
	detector.feature_variance = artifact["feature_variance"]
	from src.ood.ood_detector import OODThresholds
	detector.thresholds = OODThresholds(**artifact["thresholds"])
	return detector


def predict_image(image: Any) -> PredictionResult:
	"""Guard, validate, and classify one image using the saved checkpoint."""

	try:
		guarded = prepare_guarded_image(image)
	except ValueError as error:
		return {
			"predicted_class": "UNSUPPORTED",
			"confidence": None,
			"probabilities": {},
			"message": str(error),
			"ood": None,
		}
	try:
		detector = load_calibrated_ood_detector()
		ood_result = detector.validate_tensor(guarded.tensor)
	except Exception as error:
		return {"predicted_class": "UNSUPPORTED", "confidence": None,
			"probabilities": {}, "message": f"OOD validation unavailable: {error}", "ood": None}
	if not ood_result["accepted"]:
		return {
			"predicted_class": "UNSUPPORTED",
			"confidence": None,
			"probabilities": {},
			"message": (
				"This image appears to be outside the supported radio-galaxy input domain. "
				"The FR-I/FR-II classifier was not applied."
			),
			"ood": ood_result,
		}

	try:
		with torch.no_grad():
			probabilities_tensor = torch.softmax(detector.model(guarded.tensor), dim=1)[0]
		if not torch.isfinite(probabilities_tensor).all():
			raise ValueError("Non-finite classifier output.")
	except Exception as error:
		return {"predicted_class": "UNSUPPORTED", "confidence": None,
			"probabilities": {}, "message": f"Inference unavailable: {error}", "ood": ood_result}
	probabilities = {
		CLASS_NAMES[index]: float(probabilities_tensor[index])
		for index in range(len(CLASS_NAMES))
	}
	predicted_class = max(probabilities, key=probabilities.get)
	return {
		"predicted_class": predicted_class,
		"confidence": probabilities[predicted_class],
		"probabilities": probabilities,
		"message": "Image accepted as in-distribution for FR-I/FR-II classification.",
		"ood": ood_result,
	}