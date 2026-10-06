"""Export two original MiraBest held-out images for manual testing, never tuning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.data.download_mirabest import load_mirabest


LABEL_NAMES = {0: "FR-I", 1: "FR-II"}
EXPORT_NAMES = {
	0: "mirabest_fri.png",
	1: "mirabest_frii.png",
}


def export_samples(config_path: Path, output_dir: Path) -> list[dict[str, object]]:
	config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
	dataset_root = PROJECT_ROOT / config["paths"]["mirabest_root"]
	split_path = PROJECT_ROOT / config["paths"]["splits"] / "split_indices.json"
	if not split_path.is_file():
		raise FileNotFoundError(f"Saved split indices not found: {split_path}")

	split_indices = json.loads(split_path.read_text(encoding="utf-8"))
	dataset = load_mirabest(dataset_root, download=False)
	selected: dict[int, int] = {}
	for dataset_index in split_indices["test"]:
		_, label = dataset[dataset_index]
		label = int(label)
		if label in LABEL_NAMES and label not in selected:
			selected[label] = dataset_index
		if len(selected) == 2:
			break

	if len(selected) != 2:
		raise ValueError("The saved test split does not contain a sample for both classes.")

	output_dir.mkdir(parents=True, exist_ok=True)
	rows: list[dict[str, object]] = []
	for label, filename in EXPORT_NAMES.items():
		dataset_index = selected[label]
		image, true_label = dataset[dataset_index]
		if not isinstance(image, Image.Image) or image.mode != "L":
			raise TypeError("The verified MiraBest loader must return original grayscale PIL images.")
		if image.size != (150, 150):
			raise ValueError(f"Unexpected image dimensions for dataset index {dataset_index}: {image.size}")
		output_path = output_dir / filename
		# Lossless encoding only: no augmentation, resizing, or intensity changes.
		image.save(output_path, format="PNG")
		with Image.open(output_path) as exported:
			if exported.mode != image.mode or exported.size != image.size or exported.tobytes() != image.tobytes():
				raise ValueError(f"PNG export did not preserve original pixels: {output_path}")
		rows.append({"filename": filename, "true_label": LABEL_NAMES[int(true_label)],
			"label": int(true_label), "dataset_index": dataset_index,
			"output_path": str(output_path.resolve())})
	return rows


def main() -> int:
	parser = argparse.ArgumentParser(description="Export one original held-out MiraBest image per class for manual testing only.")
	parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
	parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "demo_samples")
	args = parser.parse_args()
	rows = export_samples(args.config, args.output_dir)
	for row in rows:
		print(f"dataset index: {row['dataset_index']} | true label: {row['label']} ({row['true_label']}) | output path: {row['output_path']}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
