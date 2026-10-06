"""AUTO behavior, synthetic rejection, explicit regressions, and split discipline."""

import hashlib
import io
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
from PIL import Image
import pytest
import torch

from src.inference import predict
from src.inference import auto
from src.ood.auto_radio import extract_auto_candidates, standardize_grayscale
from src.ood.calibration import checkpoint_sha256
from src.ood.external_compatibility import (
    EXTERNAL_CALIBRATION_PATH, PositiveVariantDataset, TRANSFORMATIONS,
    calibrate_external_from_splits, external_style_variants,
    load_external_detector, save_external_calibration, validate_external_artifact,
)
from src.ood.input_guard import prepare_guarded_image
from src.ood.ood_detector import OODThresholds

ROOT = Path(__file__).resolve().parents[1]


def synthetic_map(size=(150, 150), background=0):
    width, height = size
    y, x = np.mgrid[:height, :width]
    pixels = np.full((height, width), background, dtype=float)
    for fraction in (0.12, 0.88):
        cx, cy = (width * fraction, height / 2) if width >= height else (width / 2, height * fraction)
        pixels += 210 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 40)
    return Image.fromarray(np.rint(pixels).clip(0, 255).astype(np.uint8))


def color_map(image=None):
    pixels = np.asarray(image or synthetic_map(), dtype=float)
    return Image.fromarray(np.stack((pixels, np.rint(pixels * 0.75), np.rint(pixels * 0.5)), axis=-1).astype(np.uint8))


def detector_stub(accepted=True, logits=(2.0, 0.0)):
    detector = Mock()
    detector.validate_tensor.return_value = {"accepted": accepted, "reason": "stub gate"}
    detector.model.return_value = torch.tensor([logits])
    return detector


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_compatible_synthetic_pixels_are_preserved(mode):
    image = synthetic_map().convert(mode)
    prepared = extract_auto_candidates(image)
    assert prepared.already_compatible
    assert len(prepared.candidates) == 1
    assert prepared.candidates[0].guarded.image.tobytes() == image.convert("L").tobytes()
    assert torch.equal(prepared.candidates[0].guarded.tensor, prepare_guarded_image(image).tensor)


@pytest.mark.parametrize("shape", [(180, 80), (80, 180)])
def test_rectangular_full_frame_and_background_padding(shape):
    image = synthetic_map(shape, background=12)
    first = extract_auto_candidates(image)
    second = extract_auto_candidates(image)
    assert not first.already_compatible
    guarded = first.candidates[0].guarded
    assert torch.equal(guarded.tensor, second.candidates[0].guarded.tensor)
    square = Image.new("L", (180, 180), 12)
    square.paste(image, ((180 - shape[0]) // 2, (180 - shape[1]) // 2))
    expected = square.resize((150, 150), Image.Resampling.BILINEAR)
    assert guarded.image.tobytes() == expected.tobytes()
    if shape[0] > shape[1]:
        assert np.asarray(guarded.image)[:, :30].max() > 150
        assert np.asarray(guarded.image)[:, -30:].max() > 150
    else:
        assert np.asarray(guarded.image)[:30].max() > 150
        assert np.asarray(guarded.image)[-30:].max() > 150


def test_even_edge_structures_are_padded_not_cropped():
    pixels = np.asarray(synthetic_map((180, 80))).copy()
    pixels[38:42, 0:3] = 180
    pixels[38:42, -3:] = 180
    image = Image.fromarray(pixels)
    prepared = extract_auto_candidates(image).candidates[0].guarded.image
    assert np.asarray(prepared)[:, 0].max() > 100
    assert np.asarray(prepared)[:, -1].max() > 100


def test_colorized_map_candidates_have_no_intensity_stretch():
    image = color_map()
    prepared = extract_auto_candidates(image)
    assert not prepared.already_compatible
    assert {c.representation for c in prepared.candidates} == {"luminance", "R_channel", "G_channel", "B_channel"}
    sources = {"luminance": image.convert("L"), **{c + "_channel": image.getchannel(c) for c in "RGB"}}
    for candidate in prepared.candidates:
        assert candidate.guarded.image.tobytes() == sources[candidate.representation].tobytes()


def test_independent_sparse_channels_cannot_hide_mismatch_in_empty_background():
    pixels = np.asarray(synthetic_map()).copy()
    other = np.roll(pixels, 12, axis=0)
    image = Image.fromarray(np.stack((pixels, other, pixels), axis=-1))
    with pytest.raises(ValueError, match="inconsistent structures"):
        extract_auto_candidates(image)


def test_palette_png_handling_and_transparency():
    image = synthetic_map().convert("P")
    assert extract_auto_candidates(image).candidates
    image.info["transparency"] = 0
    with pytest.raises(ValueError, match="Transparent"):
        extract_auto_candidates(image)


@pytest.mark.parametrize("invalid", [None, b"corrupt", Image.new("L", (150, 150)), Image.new("L", (150, 150), 128)])
def test_invalid_inputs_rejected_before_any_detector(monkeypatch, invalid):
    loader = Mock(side_effect=AssertionError("Must not load model"))
    monkeypatch.setattr(predict, "load_calibrated_ood_detector", loader)
    result = predict.predict_image(invalid, input_mode="auto_web")
    assert result["predicted_class"] == "UNSUPPORTED"
    assert result["probabilities"] == {}
    assert result["message"]
    loader.assert_not_called()


def test_near_uniform_input():
    pixels = np.full((150, 150), 128, dtype=np.uint8)
    pixels[75, 75] = 129
    with pytest.raises(ValueError, match="Near-uniform"):
        extract_auto_candidates(Image.fromarray(pixels))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_values_checked_before_conversion(value):
    pixels = np.asarray(synthetic_map(), dtype=np.float32).copy()
    pixels[0, 0] = value
    with pytest.raises(ValueError, match="non-finite"):
        extract_auto_candidates(Image.fromarray(pixels))


def test_transparent_and_malformed_png():
    image = synthetic_map().convert("RGBA")
    image.putalpha(254)
    with pytest.raises(ValueError, match="Transparent"):
        extract_auto_candidates(image)
    buffer = io.BytesIO()
    synthetic_map().save(buffer, format="PNG")
    damaged = Image.open(io.BytesIO(buffer.getvalue()[:60]))
    with pytest.raises(ValueError, match="malformed|corrupted"):
        extract_auto_candidates(damaged)


@pytest.mark.parametrize("kind", ["dense_photo_style", "sparse_channel_dense_others", "independent_color_structure", "screenshot", "high_frequency"])
def test_unrelated_image_guard_cannot_be_bypassed_by_one_channel(monkeypatch, kind):
    rng = np.random.default_rng(42)
    if kind == "dense_photo_style":
        image = Image.fromarray(rng.integers(0, 256, (150, 150), dtype=np.uint8))
    elif kind == "sparse_channel_dense_others":
        sparse = np.asarray(synthetic_map())
        dense = rng.integers(0, 256, (150, 150), dtype=np.uint8)
        image = Image.fromarray(np.stack((sparse, dense, dense), axis=-1))
    elif kind == "independent_color_structure":
        first = np.asarray(synthetic_map())
        second = np.roll(first, 25, axis=0)
        image = Image.fromarray(np.stack((first, second, first), axis=-1))
    elif kind == "screenshot":
        pixels = np.full((150, 150), 255, dtype=np.uint8)
        pixels[10:140:10, 10:140] = 0
        image = Image.fromarray(pixels)
    else:
        pixels = np.zeros((150, 150), dtype=np.uint8)
        pixels[40:80:2, 40:80:2] = 255
        image = Image.fromarray(pixels)
    loader = Mock(return_value=detector_stub())
    monkeypatch.setattr(predict, "load_calibrated_ood_detector", loader)
    monkeypatch.setattr(auto, "load_external_detector", Mock(return_value=detector_stub()))
    result = predict.predict_image(image, input_mode="auto_web")
    assert result["predicted_class"] == "UNSUPPORTED"
    loader.assert_not_called()


def test_ambiguity_is_not_resolved_by_highest_confidence(monkeypatch):
    original = detector_stub(False)
    external = detector_stub()
    external.model.side_effect = [torch.tensor([[9., 0.]]), torch.tensor([[0., 2.]]),
                                  torch.tensor([[3., 0.]]), torch.tensor([[3., 0.]])]
    monkeypatch.setattr(predict, "load_calibrated_ood_detector", lambda: original)
    monkeypatch.setattr(auto, "load_external_detector", lambda *a: external)
    result = predict.predict_image(color_map(), input_mode="auto_web")
    assert result["predicted_class"] == "AMBIGUOUS"
    assert result["confidence"] is None and result["probabilities"] == {}
    assert "inconsistent" in result["message"]
    assert external.model.call_count == 4


@pytest.mark.parametrize("failure", ["rejected_candidate", "low_confidence", "missing_calibration", "bad_logits"])
def test_external_gate_failures_fail_closed(monkeypatch, failure):
    original = detector_stub(False)
    external = detector_stub()
    if failure == "rejected_candidate":
        external.validate_tensor.side_effect = [{"accepted": True}, {"accepted": False},
                                                {"accepted": True}, {"accepted": True}]
    elif failure == "low_confidence":
        external.model.return_value = torch.tensor([[0.1, 0.0]])
    elif failure == "bad_logits":
        external.model.return_value = torch.tensor([[float("nan"), 0.]])
    monkeypatch.setattr(predict, "load_calibrated_ood_detector", lambda: original)
    loader = Mock(return_value=external)
    if failure == "missing_calibration":
        loader.side_effect = FileNotFoundError("external calibration missing")
    monkeypatch.setattr(auto, "load_external_detector", loader)
    result = predict.predict_image(color_map(), input_mode="auto_web")
    assert result["predicted_class"] == "UNSUPPORTED"
    assert result["probabilities"] == {}


def test_external_averages_all_agreeing_candidates(monkeypatch):
    original = detector_stub(False)
    external = detector_stub()
    logits = [torch.tensor([[v, 0.]]) for v in (2., 3., 4., 5.)]
    external.model.side_effect = logits
    monkeypatch.setattr(predict, "load_calibrated_ood_detector", lambda: original)
    monkeypatch.setattr(auto, "load_external_detector", lambda *a: external)
    result = predict.predict_image(color_map(), input_mode="auto_web")
    assert result["predicted_class"] == "FRI"
    expected = torch.stack([torch.softmax(value, dim=1)[0] for value in logits]).mean(dim=0)
    assert result["confidence"] == pytest.approx(float(expected[0]))
    assert len(result["representations"]) == 4


@pytest.mark.parametrize("name,label", [("fri", "FRI"), ("frii", "FRII")])
@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_requested_demo_regression_exact_model_tensors(monkeypatch, name, label, mode):
    with Image.open(ROOT / f"data/demo_samples/mirabest_{name}.png") as exported:
        image = exported.convert(mode)
    detector = predict.load_calibrated_ood_detector()
    monkeypatch.setattr(predict, "load_calibrated_ood_detector", lambda: detector)
    monkeypatch.setattr(auto, "load_external_detector", Mock(side_effect=AssertionError("Compatible regression must keep original OOD")))
    captured = []
    hook = detector.model.register_forward_pre_hook(lambda model, args: captured.append(args[0].clone()))
    try:
        original = predict.predict_image(image)
        automatic = predict.predict_image(image, input_mode="auto_web")
    finally:
        hook.remove()
    assert original["predicted_class"] == automatic["predicted_class"] == label
    assert original["probabilities"] == automatic["probabilities"]
    assert original["ood"] == automatic["ood"]
    assert len(captured) == 2 and torch.equal(captured[0], captured[1])


def test_fixed_variants_are_deterministic_and_complete():
    image = synthetic_map()
    first = list(external_style_variants(image))
    second = list(external_style_variants(image))
    assert {name for name, _ in first} == set(TRANSFORMATIONS)
    assert [(name, variant.tobytes()) for name, variant in first] == [(name, variant.tobytes()) for name, variant in second]


def test_calibration_accesses_train_and_validation_only(monkeypatch):
    from src.ood import external_compatibility as calibration
    visited = []
    class Dataset:
        def __getitem__(self, index):
            assert index != 2, "Final TEST sample accessed"
            visited.append(index)
            return synthetic_map(), 0
    detector = Mock()
    detector.fit_feature_statistics.side_effect = lambda view: view[0]
    detector.calibrate.side_effect = lambda view, percentile: view[0]
    monkeypatch.setattr(calibration, "OODDetector", lambda *a, **kw: detector)
    calibrate_external_from_splits(Mock(), Dataset(), {"train": [0], "validation": [1], "test": [2]})
    assert visited == [0, 1]
    detector.fit_feature_statistics.assert_called_once()
    detector.calibrate.assert_called_once()


@pytest.fixture
def external_artifact(tmp_path):
    checkpoint = tmp_path / "locked.pth"
    checkpoint.write_bytes(b"locked test checkpoint")
    splits = {"train": [0], "validation": [1], "test": [2]}
    split_path = tmp_path / "splits.json"
    split_path.write_text(json.dumps(splits))
    detector = Mock(feature_mean=torch.zeros(512), feature_variance=torch.ones(512),
                    thresholds=OODThresholds(-0.5, 50, 0.99))
    path = tmp_path / "external_compatibility_calibration.pt"
    save_external_calibration(detector, path, checkpoint, split_path, splits, {}, seed=42)
    return path, checkpoint, split_path


@pytest.mark.parametrize("change", ["checkpoint", "split_hash", "test_indices", "statistics", "policy", "thresholds", "indices"])
def test_stale_or_invalid_external_calibration_rejected(external_artifact, change):
    path, checkpoint, splits = external_artifact
    artifact = torch.load(path, weights_only=True)
    if change == "checkpoint":
        checkpoint.write_bytes(b"changed")
    elif change == "split_hash":
        splits.write_bytes(splits.read_bytes() + b"\n")
    elif change == "test_indices":
        artifact["splits"]["test"] = {"indices": [2], "count": 1}
    elif change == "statistics":
        artifact["feature_variance"][0] = 0
    elif change == "policy":
        artifact["preprocessing_version"] = "changed"
    elif change == "thresholds":
        artifact["thresholds"]["feature_distance_threshold"] = float("nan")
    else:
        artifact["splits"]["train"]["indices"] = [2]
    with pytest.raises(ValueError):
        validate_external_artifact(artifact, checkpoint)


def test_isolated_output_filename_enforced(external_artifact):
    path, checkpoint, split_path = external_artifact
    with pytest.raises(ValueError, match="isolated"):
        save_external_calibration(Mock(), path.with_name("ood_calibration.pt"), checkpoint,
                                  split_path, json.loads(split_path.read_bytes()), {})


def test_generated_calibration_has_train_validation_provenance_only():
    artifact = torch.load(EXTERNAL_CALIBRATION_PATH, weights_only=True)
    splits = json.loads((ROOT / "data/splits/split_indices.json").read_bytes())
    assert set(artifact["splits"]) == {"train", "validation"}
    for name in ("train", "validation"):
        assert artifact["splits"][name]["indices"] == splits[name]
        assert set(artifact["splits"][name]["indices"]).isdisjoint(splits["test"])
    assert artifact["feature_dimension"] == 512
    assert artifact["checkpoint_sha256"] == checkpoint_sha256(predict.CHECKPOINT_PATH)
    assert artifact["seed"] == 42
    assert artifact["transformations"] == TRANSFORMATIONS
    validate_external_artifact(artifact, predict.CHECKPOINT_PATH)


def test_original_ood_artifact_unchanged_during_auto_prediction():
    paths = [predict.CALIBRATION_PATH, predict.CALIBRATION_PATH.with_suffix(".json")]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    original = predict.load_calibrated_ood_detector().thresholds
    predict.predict_image(synthetic_map((180, 80)), input_mode="auto_web")
    assert predict.load_calibrated_ood_detector().thresholds == original
    assert before == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def test_ui_default_is_auto_and_rejection_and_ambiguity_have_reasons(monkeypatch):
    from demo import app
    demo = app.create_demo()
    assert demo.input_components[0].image_mode is None  # Alpha must reach our guard.
    assert demo.input_components[1].value == "auto_web"
    assert len(demo.output_components) == 3
    for status in ("UNSUPPORTED", "AMBIGUOUS"):
        monkeypatch.setattr(app, "predict_image", lambda *a, **kw: {
            "predicted_class": status, "message": "useful reason", "probabilities": {}})
        _, label, reason = app.predict_for_ui_with_details(synthetic_map())
        assert status in label and reason == "useful reason"


def test_gradio_upload_preserves_alpha_for_auto_rejection(tmp_path):
    from demo.app import create_demo
    from gradio.data_classes import ImageData
    image = synthetic_map().convert("RGBA")
    image.putalpha(100)
    path = tmp_path / "transparent.png"
    image.save(path)
    uploaded = create_demo().input_components[0].preprocess(ImageData(path=str(path)))
    assert uploaded.mode == "RGBA"
    result = predict.predict_image(uploaded, input_mode="auto_web")
    assert result["predicted_class"] == "UNSUPPORTED"
    assert "Transparent" in result["message"]
