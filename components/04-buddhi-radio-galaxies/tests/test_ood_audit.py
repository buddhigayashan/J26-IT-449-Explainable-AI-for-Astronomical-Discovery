"""Audit regressions using synthetic images/features only, never final samples."""

from unittest.mock import Mock
import json

import numpy as np
from PIL import Image
import pytest
import torch

from scripts.run_ood_evaluation import scan_ood_images, score_metrics
from scripts.calibrate_ood_detector import calibrate_from_splits, deterministic_transform
from src.inference import predict
from src.ood.calibration import save_calibration, load_calibration, validate_splits
from src.ood.ood_detector import OODDetector, OODThresholds
from src.ood.input_guard import prepare_guarded_image


def valid_image():
    return Image.fromarray(np.tile(np.arange(150, dtype=np.uint8), (150, 1)))


def low_variance_image():
    pixels = np.full((150, 150), 128, dtype=np.uint8)
    pixels[75, 75] = 129
    return Image.fromarray(pixels)


@pytest.mark.parametrize("kind", ["low_variance", "blank", "rgb"])
def test_trusted_calibration_preprocessing_preserves_low_variance_samples(kind):
    image = low_variance_image()
    if kind == "blank":
        image = Image.new("L", (150, 150), 0)
    elif kind == "rgb":
        image = image.convert("RGB").resize((80, 120))
    tensor = deterministic_transform(image)
    expected_pixels = np.asarray(image.convert("L").resize(
        (150, 150), Image.Resampling.BILINEAR), dtype=np.float32)
    expected = torch.from_numpy(expected_pixels).unsqueeze(0) / 255.0
    expected = (expected - 0.5) / 0.5
    assert tensor.shape == (1, 150, 150)
    assert tensor.dtype == torch.float32
    assert torch.equal(tensor, expected)
    assert torch.equal(tensor, deterministic_transform(image))


def test_same_low_variance_external_upload_remains_rejected():
    with pytest.raises(ValueError, match="Near-uniform"):
        prepare_guarded_image(low_variance_image())


def test_scan_records_two_valid_and_malformed_blank_files(tmp_path):
    valid_image().save(tmp_path / "person.png")
    valid_image().convert("RGB").save(tmp_path / "landscape.jpg")
    Image.new("L", (150, 150)).save(tmp_path / "blank.png")
    (tmp_path / "malformed.png").write_bytes(b"not an image")
    detector = Mock()
    detector.validate_tensor.return_value = {
        "accepted": True, "ood_status": "in_distribution", "reason": "accepted",
        "energy_score": -2.0, "feature_distance": 1.0,
    }
    records = scan_ood_images(tmp_path, detector)
    assert len(records) == 4
    assert len({r["sample_id"] for r in records}) == 4
    assert detector.validate_tensor.call_count == 2
    rejected = [r for r in records if not r["accepted"]]
    assert len(rejected) == 2
    assert all(r["energy_score"] is None and r["reason"] for r in rejected)


@pytest.mark.parametrize("field,id_scores,ood_scores", [
    ("energy_score", [-10, -9], [-3, -2]),
    ("feature_distance", [1, 2], [8, 9]),
])
def test_score_direction(field, id_scores, ood_scores):
    records = [{"source": source, field: value}
               for source, scores in (("in_distribution", id_scores), ("ood", ood_scores))
               for value in scores]
    records.append({"source": "ood", field: None})
    metrics = score_metrics(records, field)
    assert metrics["auroc"] == 1
    assert metrics["fpr_at_95_tpr"] == 0
    assert metrics["excluded_count"] == 1


def test_calibration_reads_only_train_and_validation(monkeypatch):
    from scripts import calibrate_ood_detector as script
    visited = []

    class Dataset:
        def __getitem__(self, index):
            assert index != 2, "Final test sample accessed"
            visited.append(index)
            return low_variance_image(), 0

    detector = Mock()
    detector.fit_feature_statistics.side_effect = lambda view: view[0]
    detector.calibrate.side_effect = lambda view, percentile: view[0]
    monkeypatch.setattr(script, "OODDetector", lambda *a, **kw: detector)
    calibrate_from_splits(Mock(), Dataset(), {"train": [0], "validation": [1], "test": [2]})
    assert visited == [0, 1]
    assert detector.fit_feature_statistics.call_args.args[0].indices == [0]
    assert detector.calibrate.call_args.args[0].indices == [1]
    with pytest.raises(ValueError, match="overlap"):
        validate_splits({"train": [0, 2], "validation": [1], "test": [2]})


@pytest.fixture
def frozen(tmp_path):
    checkpoint = tmp_path / "locked.pth"
    checkpoint.write_bytes(b"locked checkpoint")
    splits = {"train": [0], "validation": [1], "test": [2]}
    split_path = tmp_path / "splits.json"
    split_path.write_text(json.dumps(splits))
    detector = OODDetector(Mock())
    detector.feature_mean = torch.zeros(512)
    detector.feature_variance = torch.ones(512)
    detector.thresholds = OODThresholds(-2, 10, 0.99)
    path = tmp_path / "calibration.pt"
    save_calibration(detector, path, checkpoint, split_path, splits)
    return path, checkpoint


def test_frozen_round_trip(frozen):
    path, checkpoint = frozen
    artifact = load_calibration(path, checkpoint)
    assert torch.equal(artifact["feature_mean"], torch.zeros(512))
    assert torch.equal(artifact["feature_variance"], torch.ones(512))
    assert artifact["thresholds"]["energy_threshold"] == -2
    assert artifact["splits"]["train"]["count"] == 1
    assert path.with_suffix(".json").is_file()


@pytest.mark.parametrize("split_name,index", [("test", 2), ("validation", 1)])
def test_evaluation_trusted_low_variance_uses_frozen_calibration(
        frozen, tmp_path, monkeypatch, split_name, index):
    from scripts import run_ood_evaluation as evaluation

    path, checkpoint = frozen
    artifact_bytes = path.read_bytes()
    artifact = load_calibration(path, checkpoint)
    monkeypatch.setattr(predict, "CALIBRATION_PATH", path)
    monkeypatch.setattr(predict, "CHECKPOINT_PATH", checkpoint)
    monkeypatch.setattr(predict, "load_checkpoint", Mock(return_value=Mock()))
    frozen_loader = Mock(wraps=predict.load_calibration)
    monkeypatch.setattr(predict, "load_calibration", frozen_loader)
    monkeypatch.setattr(OODDetector, "fit_feature_statistics",
                        Mock(side_effect=AssertionError("Evaluation must not fit statistics")))
    monkeypatch.setattr(OODDetector, "calibrate",
                        Mock(side_effect=AssertionError("Evaluation must not recalibrate")))
    tensors = []

    def validate(detector, tensor):
        assert detector.thresholds == OODThresholds(**artifact["thresholds"])
        assert torch.equal(detector.feature_mean, artifact["feature_mean"])
        assert torch.equal(detector.feature_variance, artifact["feature_variance"])
        tensors.append(tensor)
        return {"accepted": True, "ood_status": "in_distribution", "reason": "accepted",
                "energy_score": -3.0, "feature_distance": 1.0}

    monkeypatch.setattr(OODDetector, "validate_tensor", validate)
    visited = []

    class SyntheticDataset:
        def __getitem__(self, sample_index):
            assert sample_index == index, "Unexpected train/other split access"
            visited.append(sample_index)
            return low_variance_image(), 0

    monkeypatch.setattr(evaluation, "load_mirabest", lambda *a, **kw: SyntheticDataset())
    monkeypatch.setattr(evaluation, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(evaluation, "CALIBRATION_PATH", path)
    monkeypatch.setattr(evaluation, "_plot_scores", lambda *a: None)
    split_dir = tmp_path / "saved_splits"
    split_dir.mkdir()
    (split_dir / "split_indices.json").write_bytes((tmp_path / "splits.json").read_bytes())
    config = tmp_path / "config.yaml"
    config.write_text('paths:\n  splits: saved_splits\n  mirabest_root: unused\n')
    external_dir = tmp_path / "external"
    external_dir.mkdir()
    low_variance_image().save(external_dir / "person.png")

    summary = evaluation.run_ood_evaluation(config, tmp_path / "results", external_dir, split_name)

    frozen_loader.assert_called_once_with(path, checkpoint)
    assert visited == [index]
    assert len(tensors) == 1  # External near-uniform image was guard-rejected before scoring.
    assert torch.equal(tensors[0], deterministic_transform(low_variance_image()).unsqueeze(0))
    assert summary["in_distribution"]["accepted"] == 1
    assert summary["ood"]["rejected"] == 1
    assert summary["threshold_calibration"]["energy_threshold"] == artifact["thresholds"]["energy_threshold"]
    assert summary["threshold_calibration"]["feature_distance_threshold"] == artifact["thresholds"]["feature_distance_threshold"]
    assert summary["threshold_calibration"]["validation_percentile"] == artifact["thresholds"]["validation_percentile"]
    assert path.read_bytes() == artifact_bytes


@pytest.mark.parametrize("change", ["modified", "missing", "corrupted"])
def test_current_split_file_failures_fail_closed(frozen, monkeypatch, change):
    path, checkpoint = frozen
    split_path = path.parent / "splits.json"
    if change == "missing":
        split_path.unlink()
    elif change == "corrupted":
        split_path.write_bytes(b"not JSON")
    else:
        # Even whitespace-only changes must fail the saved SHA-256 check.
        split_path.write_bytes(split_path.read_bytes() + b"\n")
    with pytest.raises((ValueError, OSError)):
        load_calibration(path, checkpoint)
    monkeypatch.setattr(predict, "CALIBRATION_PATH", path)
    monkeypatch.setattr(predict, "CHECKPOINT_PATH", checkpoint)
    assert_unsupported(predict.predict_image(valid_image()))


@pytest.mark.parametrize("name", ["train", "validation"])
def test_changed_saved_indices_rejected_against_current_split(frozen, name):
    path, checkpoint = frozen
    artifact = load_calibration(path, checkpoint)
    artifact["splits"][name]["indices"] = [3]
    torch.save(artifact, path)
    with pytest.raises(ValueError, match=f"{name} indices mismatch"):
        load_calibration(path, checkpoint)


@pytest.mark.parametrize("name", ["train", "validation"])
def test_changed_current_indices_rejected_even_with_matching_hash(frozen, name):
    path, checkpoint = frozen
    artifact = load_calibration(path, checkpoint)
    split_path = path.parent / "splits.json"
    splits = json.loads(split_path.read_bytes())
    splits[name] = [3]
    split_path.write_text(json.dumps(splits))
    # Exercise the index comparison independently of the hash check.
    from src.ood.calibration import checkpoint_sha256
    for split in artifact["splits"].values():
        split["reference_sha256"] = checkpoint_sha256(split_path)
    torch.save(artifact, path)
    with pytest.raises(ValueError, match=f"{name} indices mismatch"):
        load_calibration(path, checkpoint)


def test_inconsistent_split_references_rejected(frozen):
    path, checkpoint = frozen
    artifact = load_calibration(path, checkpoint)
    artifact["splits"]["validation"]["reference"] = str(path.parent / "other.json")
    torch.save(artifact, path)
    with pytest.raises(ValueError, match="split reference mismatch"):
        load_calibration(path, checkpoint)


def assert_unsupported(result):
    assert result["predicted_class"] == "UNSUPPORTED"
    assert result["confidence"] is None
    assert result["probabilities"] == {}
    assert result["message"]


def test_hash_mismatch_fails_closed(frozen, monkeypatch):
    path, checkpoint = frozen
    checkpoint.write_bytes(b"changed")
    monkeypatch.setattr(predict, "CALIBRATION_PATH", path)
    monkeypatch.setattr(predict, "CHECKPOINT_PATH", checkpoint)
    result = predict.predict_image(valid_image())
    assert_unsupported(result)
    assert "hash mismatch" in result["message"]


@pytest.mark.parametrize("kind", ["missing", "malformed", "invalid_variance"])
def test_bad_artifact_fails_closed(frozen, monkeypatch, kind):
    path, checkpoint = frozen
    if kind == "missing":
        path = path.with_name("missing.pt")
    elif kind == "malformed":
        path.write_bytes(b"bad")
    else:
        artifact = load_calibration(path, checkpoint)
        artifact["feature_variance"][0] = 0
        torch.save(artifact, path)
    monkeypatch.setattr(predict, "CALIBRATION_PATH", path)
    monkeypatch.setattr(predict, "CHECKPOINT_PATH", checkpoint)
    assert_unsupported(predict.predict_image(valid_image()))


@pytest.mark.parametrize("failure", ["checkpoint", "detector", "features", "rejection"])
def test_errors_and_rejection_have_no_probabilities(frozen, monkeypatch, failure):
    path, checkpoint = frozen
    monkeypatch.setattr(predict, "CALIBRATION_PATH", path)
    monkeypatch.setattr(predict, "CHECKPOINT_PATH", checkpoint)
    if failure == "checkpoint":
        loader = Mock(side_effect=RuntimeError("checkpoint loading failed"))
        monkeypatch.setattr(predict, "load_checkpoint", loader)
    elif failure == "detector":
        monkeypatch.setattr(predict, "load_calibrated_ood_detector", Mock(side_effect=RuntimeError("detector loading failed")))
    else:
        detector = Mock()
        if failure == "features":
            detector.validate_tensor.side_effect = RuntimeError("feature extraction failed")
        else:
            detector.validate_tensor.return_value = {"accepted": False, "reason": "outside thresholds"}
        monkeypatch.setattr(predict, "load_calibrated_ood_detector", lambda: detector)
    assert_unsupported(predict.predict_image(valid_image()))


def test_inference_loads_frozen_without_recalibration(frozen, monkeypatch):
    path, checkpoint = frozen
    monkeypatch.setattr(predict, "CALIBRATION_PATH", path)
    monkeypatch.setattr(predict, "CHECKPOINT_PATH", checkpoint)
    loader = Mock(return_value=Mock())
    monkeypatch.setattr(predict, "load_checkpoint", loader)
    monkeypatch.setattr(OODDetector, "fit_feature_statistics", Mock(side_effect=AssertionError("runtime fitting")))
    monkeypatch.setattr(OODDetector, "calibrate", Mock(side_effect=AssertionError("runtime calibration")))
    detector = predict.load_calibrated_ood_detector()
    assert detector.thresholds.energy_threshold == -2
