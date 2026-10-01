"""Tests for controlled ResNet ablation configuration and isolation."""

import json

from scripts.run_ablation_study import (
	SEED,
	ablation_artifact_paths,
	build_ablation_views,
	build_training_config,
	get_ablation_specs,
	load_fixed_split_indices,
)
from src.data.preprocess import PreprocessConfig


def _config():
	return {
		"batch_size": 32,
		"num_workers": 0,
		"device": "cpu",
	}, {
		"learning_rate": 0.0003,
		"weight_decay": 0.0001,
		"num_epochs": 30,
		"early_stopping_patience": 7,
		"scheduler_patience": 2,
		"scheduler_factor": 0.5,
	}


def test_fixed_split_indices_are_loaded_unchanged(tmp_path):
	expected = {"train": [0], "validation": [1], "test": [2]}
	path = tmp_path / "split_indices.json"
	path.write_text(json.dumps(expected), encoding="utf-8")
	assert load_fixed_split_indices(path) == expected


def test_all_ablation_training_configs_use_seed_42():
	base_training, resnet_config = _config()
	for spec in get_ablation_specs():
		assert build_training_config(base_training, resnet_config, spec).seed == SEED == 42


def test_each_ablation_changes_only_its_intended_component():
	full, no_augmentation, no_weight_decay, no_scheduler = get_ablation_specs()
	assert no_augmentation.augmentation_enabled is False
	assert no_augmentation.weight_decay == full.weight_decay
	assert no_augmentation.scheduler_enabled is True
	assert no_weight_decay.augmentation_enabled == full.augmentation_enabled
	assert no_weight_decay.weight_decay == 0.0
	assert no_weight_decay.scheduler_enabled is True
	assert no_scheduler.augmentation_enabled == full.augmentation_enabled
	assert no_scheduler.weight_decay == full.weight_decay
	assert no_scheduler.scheduler_enabled is False


def test_ablation_views_do_not_access_test_split():
	class TestSplitSentinel:
		def __iter__(self):
			raise AssertionError("The test split must not be used by ablations.")

	class TinyDataset:
		def __len__(self):
			return 3

	split_indices = {"train": [0], "validation": [1], "test": TestSplitSentinel()}
	train_view, validation_view = build_ablation_views(
		TinyDataset(), split_indices, PreprocessConfig(), get_ablation_specs()[0]
	)
	assert train_view.indices == [0]
	assert validation_view.indices == [1]


def test_ablation_artifact_paths_are_separate_from_final_outputs(tmp_path):
	paths = [ablation_artifact_paths(tmp_path, name) for name in ("full_resnet", "no_scheduler")]
	assert paths[0]["checkpoint"] != paths[1]["checkpoint"]
	assert all(path.parent == tmp_path / "models" / "ablation" for path in (item["checkpoint"] for item in paths))
	assert all("resnet_cnn_best.pth" not in item["checkpoint"].name for item in paths)