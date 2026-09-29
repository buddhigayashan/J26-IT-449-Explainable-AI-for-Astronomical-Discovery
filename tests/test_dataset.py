"""Tests for MiraBest Zenodo v1 loading and inspection."""

from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

from src.data.download_mirabest import MiraBestFull, load_mirabest


DATASET_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "mirabest"


@pytest.fixture(scope="module")
def dataset():
	if not MiraBestFull.check_integrity(DATASET_ROOT):
		pytest.skip("MiraBest dataset is not available; tests do not trigger downloads.")
	return load_mirabest(DATASET_ROOT, download=False)


def test_dataset_loads_successfully_when_available(dataset):
	assert len(dataset) == 792


def test_validated_zenodo_v1_class_counts(dataset):
	counts = Counter(dataset[index][1] for index in range(len(dataset)))
	assert counts == {0: 351, 1: 441}


def test_samples_return_image_and_label(dataset):
	image, label = dataset[0]
	assert isinstance(image, Image.Image)
	assert isinstance(label, int)


def test_samples_have_only_confident_binary_labels(dataset):
	labels = {dataset[index][1] for index in range(len(dataset))}
	assert labels <= {0, 1}
	assert labels == {0, 1}


def test_image_dimensions_are_consistent(dataset):
	dimensions = {dataset[index][0].size for index in range(len(dataset))}
	assert dimensions == {(150, 150)}
