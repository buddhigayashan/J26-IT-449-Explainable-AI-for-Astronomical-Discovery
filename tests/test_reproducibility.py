"""Tests for the ResNet multiseed reproducibility workflow."""

import json

import pytest

from scripts.run_multiseed_resnet import (
	SEEDS,
	aggregate_results,
	build_seed_views,
	load_fixed_split_indices,
	seed_artifact_paths,
)
from src.data.preprocess import PreprocessConfig


def test_fixed_split_indices_are_identical_for_all_seeds(tmp_path):
	split_path = tmp_path / "split_indices.json"
	expected = {"train": [0, 1], "validation": [2], "test": [3]}
	split_path.write_text(json.dumps(expected), encoding="utf-8")
	loaded = [load_fixed_split_indices(split_path) for _ in SEEDS]
	assert loaded == [expected] * len(SEEDS)


def test_each_seed_has_separate_checkpoint_path(tmp_path):
	paths = [seed_artifact_paths(tmp_path, seed) for seed in SEEDS]
	checkpoints = [path_set["checkpoint"] for path_set in paths]
	assert len(set(checkpoints)) == len(SEEDS)
	assert all(path.parent == tmp_path / "models" / "reproducibility" for path in checkpoints)
	assert all("resnet_cnn_best.pth" not in path.name for path in checkpoints)


def test_seed_views_do_not_access_test_split():
	class TestSplitSentinel:
		def __iter__(self):
			raise AssertionError("The test split must not be used for training runs.")

	class TinyDataset:
		def __getitem__(self, index):
			raise AssertionError("The test cannot access dataset samples in this construction test.")

		def __len__(self):
			return 4

	split_indices = {"train": [0], "validation": [1], "test": TestSplitSentinel()}
	train_view, validation_view = build_seed_views(
		TinyDataset(), split_indices, PreprocessConfig()
	)
	assert train_view.indices == [0]
	assert validation_view.indices == [1]


def test_aggregate_mean_and_population_std_are_correct():
	results = [
		{
			"seed": 42,
			"validation_accuracy": 0.80,
			"macro_f1": 0.70,
			"balanced_accuracy": 0.75,
			"best_validation_loss": 0.40,
		},
		{
			"seed": 7,
			"validation_accuracy": 0.90,
			"macro_f1": 0.80,
			"balanced_accuracy": 0.85,
			"best_validation_loss": 0.20,
		},
	]
	summary = aggregate_results(results)
	assert summary["mean_validation_accuracy"] == pytest.approx(0.85)
	assert summary["std_validation_accuracy"] == pytest.approx(0.05)
	assert summary["mean_macro_f1"] == pytest.approx(0.75)
	assert summary["std_macro_f1"] == pytest.approx(0.05)
	assert summary["mean_balanced_accuracy"] == pytest.approx(0.80)
	assert summary["std_balanced_accuracy"] == pytest.approx(0.05)
	assert summary["mean_validation_loss"] == pytest.approx(0.30)
	assert summary["std_validation_loss"] == pytest.approx(0.10)