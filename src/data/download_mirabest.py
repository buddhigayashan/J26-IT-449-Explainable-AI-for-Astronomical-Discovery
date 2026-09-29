"""Official MiraBest loader and Zenodo v1 download helpers.

The batch layout, checksums, image shape, and binary label mapping mirror the
official ``MiraBest.py`` loader from E2CNNRadGal.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import shutil
import tarfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


DATASET_URL = "http://www.jb.man.ac.uk/research/MiraBest/full_dataset/MiraBest_full_batches.tar.gz"
ARCHIVE_NAME = "MiraBest_full_batches.tar.gz"
ARCHIVE_MD5 = "965b5daa83b9d8622bb407d718eecb51"
LOCAL_ARCHIVE_NAME = "batches.tar.gz"
LOCAL_ARCHIVE_MD5 = "fa9d6ad8fdde95b0a19df11433e8bca3"
BASE_FOLDER = "batches"

LEGACY_TRAIN_FILES = (
	("data_batch_1", "b15ae155301f316fc0b51af16b3c540d"),
	("data_batch_2", "0bf52cc1b47da591ed64127bab6df49e"),
	("data_batch_3", "98908045de6695c7b586d0bd90d78893"),
	("data_batch_4", "ec9b9b77dc019710faf1ad23f1a58a60"),
	("data_batch_5", "5190632a50830e5ec30de2973cc6b2e1"),
	("data_batch_6", "b7113d89ddd33dd179bf64cb578be78e"),
	("data_batch_7", "626c866b7610bfd08ac94ca3a17d02a1"),
)
LEGACY_TEST_FILES = (("test_batch", "5e443302dbdf3c2003d68ff9ac95f08c"),)
LEGACY_META_FILE = ("batches.meta", "e1b5450577209e583bc43fbf8e851965")
ZENODO_TRAIN_FILES = tuple((f"data_batch_{index}", "") for index in range(1, 9))
ZENODO_TEST_FILES = (("test_batch", ""),)
ZENODO_META_FILE = ("batches.meta", "")
ZENODO_MANIFEST = "hash.md5"


class MiraBestError(RuntimeError):
	"""Base exception for MiraBest loading and download failures."""


class MiraBestDownloadError(MiraBestError):
	"""Raised when the archive cannot be downloaded or extracted."""


def _md5(path: Path) -> str:
	digest = hashlib.md5()
	with path.open("rb") as file_handle:
		for block in iter(lambda: file_handle.read(1024 * 1024), b""):
			digest.update(block)
	return digest.hexdigest()


def _check_file(path: Path, expected_md5: str) -> bool:
	return path.is_file() and _md5(path) == expected_md5


def _safe_extract(archive_path: Path, destination: Path) -> None:
	destination = destination.resolve()
	with tarfile.open(archive_path, "r:gz") as archive:
		for member in archive.getmembers():
			target = (destination / member.name).resolve()
			if os.path.commonpath((str(destination), str(target))) != str(destination):
				raise MiraBestDownloadError(
					f"Unsafe path in MiraBest archive: {member.name}"
				)
		archive.extractall(destination)


class MiraBestFull(Dataset):
	"""Official MiraBest full dataset loader with optional PIL transforms."""

	def __init__(
		self,
		root: str | Path,
		train: bool = True,
		transform: Callable | None = None,
		target_transform: Callable | None = None,
	) -> None:
		self.root = Path(root).expanduser()
		self.train = train
		self.transform = transform
		self.target_transform = target_transform
		if not self.check_integrity(self.root):
			raise MiraBestError(
				f"MiraBest files are missing or corrupted under {self.root}. "
				"Run scripts/download_data.py to download the official archive."
			)

		files = self._data_files(self.root, train)
		data_parts: list[np.ndarray] = []
		self.targets: list[int] = []
		for file_name, _ in files:
			with (self.root / BASE_FOLDER / file_name).open("rb") as file_handle:
				entry = pickle.load(file_handle, encoding="latin1")
			data_parts.append(entry["data"])
			labels = entry.get("labels", entry.get("fine_labels"))
			if labels is None:
				raise MiraBestError(f"No labels found in {file_name}.")
			self.targets.extend(labels)

		self.data = np.vstack(data_parts).reshape(-1, 1, 150, 150)
		self.data = self.data.transpose((0, 2, 3, 1))
		self.classes = self._load_meta()
		self.class_to_idx = {name: index for index, name in enumerate(self.classes)}

	@classmethod
	def check_integrity(cls, root: str | Path) -> bool:
		root = Path(root).expanduser()
		batch_root = root / BASE_FOLDER
		manifest_path = batch_root / ZENODO_MANIFEST
		if manifest_path.is_file():
			try:
				manifest = {}
				for line in manifest_path.read_text(encoding="ascii").splitlines():
					checksum, name = line.split(maxsplit=1)
					manifest[name.strip()] = checksum
				return all(
					_check_file(batch_root / name, checksum)
					for name, checksum in manifest.items()
				)
			except (OSError, UnicodeError, ValueError):
				return False

		files = LEGACY_TRAIN_FILES + LEGACY_TEST_FILES + (LEGACY_META_FILE,)
		return all(_check_file(batch_root / name, checksum) for name, checksum in files)

	@classmethod
	def _data_files(cls, root: Path, train: bool) -> tuple[tuple[str, str], ...]:
		if (root / BASE_FOLDER / ZENODO_MANIFEST).is_file():
			return ZENODO_TRAIN_FILES if train else ZENODO_TEST_FILES
		return LEGACY_TRAIN_FILES if train else LEGACY_TEST_FILES

	@classmethod
	def archive_path(cls, root: str | Path) -> Path:
		root = Path(root).expanduser()
		local_archive = root / LOCAL_ARCHIVE_NAME
		return local_archive if local_archive.exists() else root / ARCHIVE_NAME

	@classmethod
	def download(cls, root: str | Path) -> Path:
		root = Path(root).expanduser()
		root.mkdir(parents=True, exist_ok=True)
		if cls.check_integrity(root):
			return root

		archive_path = cls.archive_path(root)
		if archive_path.exists():
			expected_md5 = (
				LOCAL_ARCHIVE_MD5
				if archive_path.name == LOCAL_ARCHIVE_NAME
				else ARCHIVE_MD5
			)
			if _check_file(archive_path, expected_md5):
				if (root / BASE_FOLDER).exists():
					if cls.check_integrity(root):
						return root
					raise MiraBestDownloadError(
						f"Existing {BASE_FOLDER} directory is incomplete or corrupted; "
						f"it was not altered: {root / BASE_FOLDER}"
					)
				try:
					_safe_extract(archive_path, root)
				except (OSError, tarfile.TarError) as error:
					raise MiraBestDownloadError(
						f"Could not extract existing MiraBest archive: {error}"
					) from error
				if cls.check_integrity(root):
					return root
				raise MiraBestDownloadError(
					"The MiraBest archive extracted, but one or more official "
					"checksums did not match. Original files were not altered."
				)
			raise MiraBestDownloadError(
				f"Existing archive has an unexpected checksum: {archive_path}"
			)

		temporary_path = archive_path.with_suffix(archive_path.suffix + ".part")
		try:
			with urllib.request.urlopen(DATASET_URL, timeout=60) as response, temporary_path.open(
				"wb"
			) as output:
				shutil.copyfileobj(response, output)
			if not _check_file(temporary_path, ARCHIVE_MD5):
				raise MiraBestDownloadError("Downloaded MiraBest archive checksum failed.")
			temporary_path.replace(archive_path)
			_safe_extract(archive_path, root)
		except (OSError, urllib.error.URLError, tarfile.TarError) as error:
			temporary_path.unlink(missing_ok=True)
			raise MiraBestDownloadError(f"MiraBest download failed: {error}") from error

		if not cls.check_integrity(root):
			raise MiraBestDownloadError(
				"MiraBest extraction completed, but official file checksums failed."
			)
		return root

	def _load_meta(self) -> list[str]:
		path = self.root / BASE_FOLDER / "batches.meta"
		with path.open("rb") as file_handle:
			metadata = pickle.load(file_handle, encoding="latin1")
		return metadata["label_names"]

	def __getitem__(self, index: int) -> tuple[Image.Image | torch.Tensor, int]:
		image = Image.fromarray(np.asarray(self.data[index]).reshape(150, 150), mode="L")
		target = self.targets[index]
		if self.transform is not None:
			image = self.transform(image)
		if self.target_transform is not None:
			target = self.target_transform(target)
		return image, target

	def __len__(self) -> int:
		return len(self.data)


class MiraBestBinarySubset(MiraBestFull):
	"""Binary FRI/FRII view of Zenodo v1 or a legacy full archive.

	For a legacy full archive, the official confident fine labels are retained:
	FRI labels 0, 1, 2 and FRII labels 5, 6.
	"""

	FRI_LABELS = (0, 1, 2)
	FRII_LABELS = (5, 6)
	EXCLUDED_LABELS = (3, 4, 7, 8, 9)

	def __init__(self, *args, **kwargs) -> None:
		super().__init__(*args, **kwargs)
		original_targets = np.asarray(self.targets)
		if self.classes == ["FRI", "FRII"] and set(np.unique(original_targets)) <= {0, 1}:
			self.targets = original_targets.astype(np.int64).tolist()
			self.label_mapping = {0: "FR-I", 1: "FR-II"}
			return

		keep_mask = ~np.isin(original_targets, self.EXCLUDED_LABELS)
		remapped_targets = np.full(original_targets.shape, -1, dtype=np.int64)
		remapped_targets[np.isin(original_targets, self.FRI_LABELS)] = 0
		remapped_targets[np.isin(original_targets, self.FRII_LABELS)] = 1
		self.data = self.data[keep_mask]
		self.targets = remapped_targets[keep_mask].tolist()
		self.label_mapping = {0: "FR-I", 1: "FR-II"}


class MiraBestZenodoV1(Dataset):
	"""Combined train and test MiraBest Zenodo v1 basic binary dataset."""

	label_mapping = {0: "FR-I", 1: "FR-II"}

	def __init__(self, root: str | Path) -> None:
		self.train = MiraBestBinarySubset(root=root, train=True)
		self.test = MiraBestBinarySubset(root=root, train=False)
		self._lengths = (len(self.train), len(self.test))

	def __len__(self) -> int:
		return sum(self._lengths)

	def __getitem__(self, index: int) -> tuple[Image.Image | torch.Tensor, int]:
		if index < self._lengths[0]:
			return self.train[index]
		return self.test[index - self._lengths[0]]


def load_mirabest(
	root: str | Path = "data/raw/mirabest", download: bool = False
) -> MiraBestZenodoV1:
	"""Load the MiraBest Zenodo v1 basic binary dataset without preprocessing."""

	root = Path(root).expanduser()
	if download:
		MiraBestFull.download(root)
	return MiraBestZenodoV1(root)
