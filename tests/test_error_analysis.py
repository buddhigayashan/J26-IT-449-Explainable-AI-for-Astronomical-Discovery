"""Tests for descriptive locked-model error analysis."""

import json

from PIL import Image

from scripts import run_error_analysis


def test_locked_checkpoint_path_is_used():
	assert run_error_analysis.CHECKPOINT_PATH.name == "resnet_cnn_best.pth"
	assert run_error_analysis.CHECKPOINT_PATH.parent.name == "models"


def test_exact_saved_test_split_is_loaded(tmp_path):
	split_path = tmp_path / "split_indices.json"
	expected = {"train": [1], "validation": [2], "test": [3, 4]}
	split_path.write_text(json.dumps(expected), encoding="utf-8")
	assert run_error_analysis.load_test_indices(split_path) == [3, 4]


def test_error_analysis_uses_no_augmentation():
	transform = run_error_analysis.build_inference_transform()
	transform_names = [type(operation).__name__ for operation in transform.transforms]
	assert transform_names == ["ToTensor", "Normalize"]


def test_prediction_record_probabilities_and_error_type():
	record = run_error_analysis.prediction_record(17, 0, [0.2, 0.8])
	assert record["true_label"] == "FRI"
	assert record["predicted_label"] == "FRII"
	assert record["error_type"] == "FRI_to_FRII"
	assert 0.0 <= record["probability_fri"] <= 1.0
	assert 0.0 <= record["probability_frii"] <= 1.0
	assert record["probability_fri"] + record["probability_frii"] == 1.0


def test_error_analysis_has_no_training_entry_points():
	source = run_error_analysis.__file__.read_text(encoding="utf-8") if hasattr(run_error_analysis.__file__, "read_text") else open(run_error_analysis.__file__, encoding="utf-8").read()
	assert "train_model" not in source
	assert "torch.optim" not in source


def test_output_paths_are_isolated_from_final_artifacts(tmp_path):
	paths = run_error_analysis.output_paths(tmp_path)
	assert paths["predictions"].parent == tmp_path / "error_analysis"
	assert paths["summary"].parent == tmp_path / "error_analysis"
	assert paths["predictions"].name == "test_predictions.csv"
	assert paths["summary"].name == "error_summary.json"
	assert paths["predictions"] != tmp_path / "metrics" / "resnet_test_evaluation.json"
	assert paths["summary"] != tmp_path / "metrics" / "final_test_metrics.csv"