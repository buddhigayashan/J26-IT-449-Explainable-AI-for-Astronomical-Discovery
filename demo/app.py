"""Thin Gradio UI for the selected MiraBest ResNet classifier."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predict import predict_image


DEMO_NOTE = "This prototype is designed for MiraBest-style radio galaxy images and predicts FR-I vs FR-II."


def predict_for_ui(image: Any) -> tuple[Any, dict[str, float]]:
	"""Adapt reusable inference output to Gradio image and label components."""

	result = predict_image(image)
	return image, result["probabilities"]


def create_demo():
	"""Construct the Gradio interface without launching it."""

	try:
		import gradio as gr
	except ImportError as error:
		raise RuntimeError("Gradio is not installed. Run: pip install gradio") from error

	return gr.Interface(
		fn=predict_for_ui,
		inputs=gr.Image(type="pil", label="Upload one radio galaxy image"),
		outputs=[
			gr.Image(type="pil", label="Uploaded image preview"),
			gr.Label(num_top_classes=2, label="FR-I / FR-II prediction"),
		],
		title="MiraBest FR-I / FR-II Classifier",
		description=DEMO_NOTE,
	)


if __name__ == "__main__":
	create_demo().launch()