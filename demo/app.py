"""Thin Gradio UI for the selected MiraBest ResNet classifier."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predict import predict_image


DEMO_NOTE = (
	"Supports radio-galaxy images intended for FR-I / FR-II morphology classification. "
	"Optical-only, artistic, unrelated, or strongly out-of-domain images may be rejected. "
	"Upload a radio-galaxy image from a survey, telescope source, or the web. The system "
	"attempts to recover a compatible radio-morphology representation automatically. "
	"Ambiguous inputs are rejected. External/web-image predictions are experimental and "
	"should not be treated as survey-quality classifications unless the source image contains "
	"genuine radio morphology. Valid radio maps may also be rejected; survey accuracy is unverified."
)


def predict_for_ui(image: Any, input_mode: str = "mirabest") -> tuple[Any, dict[str, float] | str]:
	"""Adapt reusable inference output to Gradio image and label components."""

	result = predict_image(image) if input_mode == "mirabest" else predict_image(image, input_mode=input_mode)
	if result["predicted_class"] in {"UNSUPPORTED", "AMBIGUOUS"}:
		return image, result["predicted_class"] + ": " + result["message"]
	return image, result["probabilities"]


def predict_for_ui_with_details(image: Any, input_mode: str = "auto_web"):
	"""Expose useful handling and scientific context without numeric gate details."""
	result = predict_image(image, input_mode=input_mode)
	label = result["predicted_class"]
	if label in {"UNSUPPORTED", "AMBIGUOUS"}:
		return image, label + ": " + result["message"], result["message"]
	display = {"FRI": "FR-I", "FRII": "FR-II"}
	scores = {display.get(key, key): value for key, value in result["probabilities"].items()}
	handling = result.get("input_handling", input_mode)
	representations = ", ".join(result.get("representations", []))
	note = f"{display.get(label, label)} | confidence: {result['confidence']:.1%} | handling: {handling}. "
	if representations:
		note += f"Representation(s): {representations}. "
	note += result["message"]
	return image, scores, note


def create_demo():
	"""Construct the Gradio interface without launching it."""

	try:
		import gradio as gr
	except ImportError as error:
		raise RuntimeError("Gradio is not installed. Run: pip install gradio") from error

	return gr.Interface(
		fn=predict_for_ui_with_details,
		inputs=[
			gr.Image(type="pil", image_mode=None, label="Upload one radio galaxy image"),
			gr.Radio(choices=[("AUTO / WEB RADIO IMAGE (EXPERIMENTAL) — recommended", "auto_web"),
				("MiraBest (original preprocessing)", "mirabest"),
				("External radio map (experimental)", "external_radio")],
				value="auto_web", label="Input preparation"),
		],
		outputs=[
			gr.Image(type="pil", label="Uploaded image preview"),
			gr.Label(num_top_classes=2, label="FR-I / FR-II, UNSUPPORTED / OUT-OF-DOMAIN, or AMBIGUOUS"),
			gr.Textbox(label="Input handling and scientific note", interactive=False),
		],
		title="MiraBest FR-I / FR-II Classifier",
		description=DEMO_NOTE,
	)


if __name__ == "__main__":
	create_demo().launch()
