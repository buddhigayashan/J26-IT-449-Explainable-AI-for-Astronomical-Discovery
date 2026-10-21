"""Export labeled MiraBest test images for the local viva demo."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from PIL import Image
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.data.download_mirabest import load_mirabest


LABEL_NAMES = {0: "FRI", 1: "FRII"}
EXPORT_NAMES = {
	0: ("FRI_test_sample_01.png", "FRI_test_sample_02.png"),
	1: ("FRII_test_sample_01.png", "FRII_test_sample_02.png"),
}


def export_samples(config_path: Path, output_dir: Path) -> list[dict[str, object]]:
	config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
	dataset_root = PROJECT_ROOT / config["paths"]["mirabest_root"]
	split_path = PROJECT_ROOT / config["paths"]["splits"] / "split_indices.json"
	if not split_path.is_file():
		raise FileNotFoundError(f"Saved split indices not found: {split_path}")

	split_indices = json.loads(split_path.read_text(encoding="utf-8"))
	dataset = load_mirabest(dataset_root, download=False)
	selected: dict[int, list[int]] = {0: [], 1: []}
	for dataset_index in split_indices["test"]:
		_, label = dataset[dataset_index]
		label = int(label)
		if label in selected and len(selected[label]) < 2:
			selected[label].append(dataset_index)

	if any(len(indices) != 2 for indices in selected.values()):
		raise ValueError("The saved test split does not contain two samples for both classes.")

	output_dir.mkdir(parents=True, exist_ok=True)
	rows: list[dict[str, object]] = []
	for label, indices in selected.items():
		for filename, dataset_index in zip(EXPORT_NAMES[label], indices):
			image, true_label = dataset[dataset_index]
			if not isinstance(image, Image.Image):
				raise TypeError("The MiraBest loader did not return a PIL image.")
			image = image.convert("L")
			if image.size != (150, 150):
				raise ValueError(f"Unexpected image dimensions for dataset index {dataset_index}: {image.size}")
			image.save(output_dir / filename, format="PNG")
			rows.append(
				{
					"filename": filename,
					"true_label": LABEL_NAMES[int(true_label)],
					"dataset_index": dataset_index,
				}
			)

	with (output_dir / "sample_labels.csv").open("w", newline="", encoding="utf-8") as file_handle:
		writer = csv.DictWriter(file_handle, fieldnames=["filename", "true_label", "dataset_index"])
		writer.writeheader()
		writer.writerows(rows)
	return rows


def main() -> int:
	parser = argparse.ArgumentParser(description="Export four labeled MiraBest test images for the viva demo.")
	parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
	parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "demo_samples")
	args = parser.parse_args()
	rows = export_samples(args.config, args.output_dir)
	print(f"Exported {len(rows)} test samples to {args.output_dir}")
	for row in rows:
		print(f"{row['filename']}: {row['true_label']} (dataset index {row['dataset_index']})")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())