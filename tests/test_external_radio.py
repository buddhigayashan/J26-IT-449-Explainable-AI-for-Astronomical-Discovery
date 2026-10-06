"""Synthetic safety checks and requested PNG consistency regressions; no tuning."""

from unittest.mock import Mock
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from src.inference import predict
from src.ood.external_radio import prepare_external_radio_image
from src.ood.input_guard import prepare_guarded_image


def radio_map(size=(180, 80)):
    width, height = size
    y, x = np.mgrid[:height, :width]
    pixels = np.zeros((height, width), dtype=np.float32)
    for center in (width * 0.12, width * 0.88):
        pixels += 230 * np.exp(-((x - center) ** 2 + (y - height / 2) ** 2) / 40)
    return Image.fromarray(pixels.astype(np.uint8))


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_external_modes_and_determinism(mode):
    image = radio_map().convert(mode)
    first = prepare_external_radio_image(image)
    second = prepare_external_radio_image(image)
    assert first.image.mode == "L"
    assert first.image.size == (150, 150)
    assert first.tensor.shape == (1, 1, 150, 150)
    assert first.tensor.dtype == torch.float32
    assert torch.isfinite(first.tensor).all()
    assert -1 <= first.tensor.min() <= first.tensor.max() <= 1
    assert torch.equal(first.tensor, second.tensor)
    assert first.image.tobytes() == second.image.tobytes()


def test_letterbox_preserves_both_off_center_lobes():
    prepared = np.asarray(prepare_external_radio_image(radio_map()).image)
    assert prepared[:40].max() == 0
    assert prepared[-40:].max() == 0
    assert prepared[:, :30].max() > 200
    assert prepared[:, -30:].max() > 200
    # Transpose must pad horizontally instead, preserving the same full frame.
    vertical = np.asarray(prepare_external_radio_image(
        radio_map().transpose(Image.Transpose.TRANSPOSE)).image)
    assert vertical[:, :40].max() == 0
    assert vertical[:, -40:].max() == 0
    assert vertical[:30].max() > 200
    assert vertical[-30:].max() > 200
    # PIL rounds each separable interpolation pass; transposing can differ by 1.
    assert np.abs(vertical.astype(int) - prepared.T.astype(int)).max() <= 1


@pytest.mark.parametrize("value", [0, 128, 255])
def test_blank_rejected(value):
    with pytest.raises(ValueError, match="uniform"):
        prepare_external_radio_image(Image.new("L", (150, 150), value))


def test_near_uniform_rejected_without_stretching():
    pixels = np.full((150, 150), 128, dtype=np.uint8)
    pixels[75, 75] = 129
    with pytest.raises(ValueError, match="Near-uniform"):
        prepare_external_radio_image(Image.fromarray(pixels))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_float_pixels_rejected(value):
    pixels = np.asarray(radio_map(), dtype=np.float32).copy()
    pixels[0, 0] = value
    with pytest.raises(ValueError, match="non-finite"):
        prepare_external_radio_image(Image.fromarray(pixels))


@pytest.mark.parametrize("image", [None, b"corrupted", np.zeros((150, 150))])
def test_invalid_external_input_fails_closed(image):
    result = predict.predict_image(image, input_mode="external_radio")
    assert result["predicted_class"] == "UNSUPPORTED"
    assert result["probabilities"] == {}
    assert result["message"]


def detector_stub(accepted=True):
    detector = Mock()
    detector.validate_tensor.return_value = {"accepted": accepted, "reason": "frozen thresholds"}
    detector.model.return_value = torch.tensor([[2.0, 1.0]])
    return detector


def test_external_radio_cannot_bypass_frozen_ood(monkeypatch):
    detector = detector_stub(False)
    monkeypatch.setattr(predict, "load_calibrated_ood_detector", lambda: detector)
    result = predict.predict_image(radio_map(), input_mode="external_radio")
    detector.validate_tensor.assert_called_once()
    detector.model.assert_not_called()
    assert result["predicted_class"] == "UNSUPPORTED"
    assert result["probabilities"] == {}
    assert "frozen thresholds" in result["message"]


def test_external_accepted_only_after_ood(monkeypatch):
    detector = detector_stub()
    monkeypatch.setattr(predict, "load_calibrated_ood_detector", lambda: detector)
    result = predict.predict_image(radio_map(), input_mode="external_radio")
    assert result["predicted_class"] == "FRI"
    assert result["ood"]["accepted"]
    assert torch.equal(detector.validate_tensor.call_args.args[0],
                       detector.model.call_args.args[0])
    assert "unverified" in result["message"]


@pytest.mark.parametrize("kind", ["dense_grayscale", "color", "screenshot", "transparent"])
def test_unrelated_styles_rejected_even_with_accepting_detector(monkeypatch, kind):
    image = radio_map().convert("RGB")
    if kind == "dense_grayscale":
        image = Image.fromarray(np.tile(np.arange(150, dtype=np.uint8), (150, 1)))
    elif kind == "color":
        pixels = np.asarray(image).copy()
        pixels[:40, :, 0] = 255
        image = Image.fromarray(pixels)
    elif kind == "screenshot":
        pixels = np.full((150, 150), 255, dtype=np.uint8)
        pixels[20:130:10, 20:130] = 0
        image = Image.fromarray(pixels)
    else:
        image = image.convert("RGBA")
        image.putalpha(128)
    loader = Mock(return_value=detector_stub())
    monkeypatch.setattr(predict, "load_calibrated_ood_detector", loader)
    result = predict.predict_image(image, input_mode="external_radio")
    assert result["predicted_class"] == "UNSUPPORTED"
    loader.assert_not_called()


def test_original_mirabest_preprocessing_and_inference_unchanged(monkeypatch):
    image = Image.fromarray(np.tile(np.arange(80, dtype=np.uint8), (120, 1)))
    detector = detector_stub()
    monkeypatch.setattr(predict, "load_calibrated_ood_detector", lambda: detector)
    result = predict.predict_image(image)
    assert result["predicted_class"] == "FRI"
    assert torch.equal(detector.validate_tensor.call_args.args[0], prepare_guarded_image(image).tensor)
    external = predict.predict_image(image, input_mode="external_radio")
    assert external["predicted_class"] == "UNSUPPORTED"


def test_external_calibration_unavailable_fails_closed(monkeypatch):
    monkeypatch.setattr(predict, "load_calibrated_ood_detector",
                        Mock(side_effect=ValueError("calibration unavailable")))
    result = predict.predict_image(radio_map(), input_mode="external_radio")
    assert result["predicted_class"] == "UNSUPPORTED"
    assert result["probabilities"] == {}


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_compatible_synthetic_map_retains_original_pixels_and_tensor(mode):
    image = radio_map((150, 150)).convert(mode)
    external = prepare_external_radio_image(image)
    original = prepare_guarded_image(image)
    assert external.image.tobytes() == image.convert("L").tobytes()
    assert torch.equal(external.tensor, original.tensor)


@pytest.mark.parametrize("mode", ["F", "I", "I;16"])
def test_numeric_maps_without_known_display_scale_fail_closed(mode):
    with pytest.raises(ValueError, match="unspecified intensity scale"):
        prepare_external_radio_image(radio_map().convert(mode))


@pytest.mark.parametrize("name,expected_label", [("fri", "FRI"), ("frii", "FRII")])
def test_exported_mirabest_png_actual_classifier_and_ood_consistency(monkeypatch, name, expected_label):
    """Read only these requested exports; never adjust gates using their outcomes."""
    path = Path(__file__).resolve().parents[1] / "data" / "demo_samples" / f"mirabest_{name}.png"
    with Image.open(path) as image:
        original = prepare_guarded_image(image)
        external = prepare_external_radio_image(image)
        assert external.image.tobytes() == image.tobytes()
        assert torch.equal(original.tensor, external.tensor)
        # Use the real locked model and frozen calibration. Capture the tensors
        # reaching model.forward as well as those passed to the OOD detector.
        detector = predict.load_calibrated_ood_detector()
        monkeypatch.setattr(predict, "load_calibrated_ood_detector", lambda: detector)
        captured = []
        hook = detector.model.register_forward_pre_hook(
            lambda model, args: captured.append(args[0].detach().clone()))
        try:
            first = predict.predict_image(image, input_mode="mirabest")
            second = predict.predict_image(image, input_mode="external_radio")
        finally:
            hook.remove()
        assert len(captured) == 2
        assert torch.equal(captured[0], original.tensor)
        assert torch.equal(captured[1], captured[0])
        assert first["predicted_class"] == second["predicted_class"] == expected_label
        assert first["probabilities"] == second["probabilities"]
        assert first["ood"] == second["ood"]
