"""Controlled experiment regressions; training is mocked and all images are synthetic."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from torchvision import transforms
import yaml

from scripts import run_rotation_augmentation_experiment as experiment


def test_only_rotation_strategy_changes():
    current, current_validation = experiment.build_experiment_transforms("current_augmentation")
    right, right_validation = experiment.build_experiment_transforms("right_angle_rotation")
    assert isinstance(current.transforms[0], transforms.RandomRotation)
    assert current.transforms[0].degrees == [-15.0, 15.0]
    assert isinstance(right.transforms[0], experiment.RandomRightAngleRotation)
    assert right.transforms[0].angles == (0, 90, 180, 270)
    assert [repr(t) for t in current.transforms[1:]] == [repr(t) for t in right.transforms[1:]]
    assert isinstance(current.transforms[1], transforms.RandomHorizontalFlip)
    assert isinstance(current.transforms[2], transforms.RandomVerticalFlip)
    assert repr(current_validation) == repr(right_validation)
    assert len(current_validation.transforms) == 2


def test_right_angle_rotation_has_only_four_exact_outputs(monkeypatch):
    pixels = np.arange(25, dtype=np.uint8).reshape(5, 5)
    image = Image.fromarray(pixels)
    rotation = experiment.RandomRightAngleRotation()
    for k in range(4):
        monkeypatch.setattr(torch, "randint", lambda *a, **kw: torch.tensor([k]))
        assert np.array_equal(np.asarray(rotation(image)), np.rot90(pixels, k))


def test_seed_fixed_and_training_settings_match_selected_resnet():
    config = yaml.safe_load(Path("config.yaml").read_text())
    config["training"]["seed"] = 99
    settings = experiment.training_settings(config)
    assert settings.seed == experiment.SEED == 42
    assert settings.batch_size == config["training"]["batch_size"]
    for field in ("learning_rate", "weight_decay", "num_epochs", "early_stopping_patience",
                  "scheduler_patience", "scheduler_factor"):
        assert getattr(settings, field) == config["models"]["resnet"][field]


def test_experiment_never_uses_test_and_preserves_final_artifacts(tmp_path, monkeypatch):
    config = yaml.safe_load(Path("config.yaml").read_text())
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    splits = {"train": [0, 1], "validation": [2, 3], "test": [4, 5]}
    split_path = tmp_path / config["paths"]["splits"] / "split_indices.json"
    split_path.parent.mkdir(parents=True)
    split_path.write_text(json.dumps(splits))
    output_root = tmp_path / "outputs"
    protected = [output_root / "models" / "resnet_cnn_best.pth",
                 output_root / "metrics" / "resnet_test_evaluation.json",
                 output_root / "metrics" / "final_test_metrics.csv"]
    for path in protected:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"existing final artifact")
    # Also check the real locked artifacts using hashes, without loading/scoring any samples.
    actual = [Path("outputs/models/resnet_cnn_best.pth"),
              Path("outputs/metrics/resnet_test_evaluation.json"),
              Path("outputs/metrics/final_test_metrics.csv")]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected + actual}
    split_before = split_path.read_bytes()
    visited = []

    class SyntheticDataset:
        def __getitem__(self, index):
            assert index not in splits["test"], "Final test sample accessed"
            visited.append(index)
            return Image.fromarray(np.arange(150 * 150, dtype=np.uint8).reshape(150, 150)), index % 2

    monkeypatch.setattr(experiment, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(experiment, "load_mirabest", lambda *a, **kw: SyntheticDataset())
    training_calls = []
    initial_weights = []

    def fake_train(train_view, validation_view, **kwargs):
        assert train_view.indices == splits["train"]
        assert validation_view.indices == splits["validation"]
        assert isinstance(kwargs["model"], experiment.ResNetCNN)
        initial_weights.append({k: v.clone() for k, v in kwargs["model"].state_dict().items()})
        training_calls.append(kwargs)
        for i in range(len(train_view)):
            train_view[i]
        for path in kwargs["artifact_paths"].values():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"mock experiment artifact")
        return {"best_epoch": 3}

    def fake_evaluate(model, checkpoint, validation_view, **kwargs):
        assert kwargs["split_name"] == "validation"
        assert validation_view.indices == splits["validation"]
        assert checkpoint.parent == output_root / "models" / "augmentation_experiment"
        for i in range(len(validation_view)):
            validation_view[i]
        return {"validation_loss": 0.4, "accuracy": 0.75, "macro_precision": 0.8,
                "macro_recall": 0.75, "macro_f1": 0.76, "balanced_accuracy": 0.75}

    monkeypatch.setattr(experiment, "train_model", fake_train)
    monkeypatch.setattr(experiment, "evaluate_model", fake_evaluate)
    results = experiment.run_experiment(config_path, output_root)
    assert [r["configuration"] for r in results] == list(experiment.STRATEGIES)
    assert all(r["seed"] == 42 and r["best_epoch"] == 3 for r in results)
    assert set(visited) == {0, 1, 2, 3}
    assert asdict(training_calls[0]["config"]) == asdict(training_calls[1]["config"])
    assert all(torch.equal(initial_weights[0][k], initial_weights[1][k]) for k in initial_weights[0])
    assert (output_root / "augmentation_experiment" / "rotation_results.csv").is_file()
    assert (output_root / "figures" / "rotation_augmentation_comparison.png").is_file()
    assert split_path.read_bytes() == split_before
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before}
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        experiment.run_experiment(config_path, output_root)
    assert len(training_calls) == 2
