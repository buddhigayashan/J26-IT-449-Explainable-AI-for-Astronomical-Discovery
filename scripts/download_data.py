"""Download, load, validate, and inspect the MiraBest Zenodo v1 dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.data.download_mirabest import MiraBestError, load_mirabest
from src.data.validate_dataset import save_sample_grid, save_validation_results, validate_dataset


def main() -> int:
	parser = argparse.ArgumentParser(
		description="Download and validate MiraBest Zenodo v1 without preprocessing or training."
	)
	parser.add_argument(
		"--root",
		type=Path,
		default=PROJECT_ROOT / "data" / "raw" / "mirabest",
		help="MiraBest dataset root (default: data/raw/mirabest)",
	)
	parser.add_argument(
		"--no-download",
		action="store_true",
		help="Load only existing official MiraBest files.",
	)
	args = parser.parse_args()

	try:
		dataset = load_mirabest(args.root, download=not args.no_download)
		results = validate_dataset(dataset)
		json_path, csv_path = save_validation_results(results, PROJECT_ROOT / "outputs" / "metrics")
		figure_path = save_sample_grid(dataset, PROJECT_ROOT / "outputs" / "figures" / "mirabest_sample_grid.png")
	except MiraBestError as error:
		print(f"MiraBest error: {error}", file=sys.stderr)
		return 1
	except (OSError, ValueError, RuntimeError) as error:
		print(f"Validation error: {error}", file=sys.stderr)
		return 1

	print("MiraBest Zenodo v1 basic binary dataset validation")
	print(f"Dataset root: {args.root}")
	print(f"Total samples: {results['total_samples']}")
	print(f"FR-I samples: {results['class_counts']['FR-I']} ({results['class_balance_percentages']['FR-I']:.2f}%)")
	print(f"FR-II samples: {results['class_counts']['FR-II']} ({results['class_balance_percentages']['FR-II']:.2f}%)")
	print(f"Image tensor shape: {results['image_tensor_shape']}")
	print(f"Image dimensions: {results['image_dimensions']}")
	print(f"Dtype: {results['dtype']}")
	print(f"Pixel range: {results['minimum_pixel_value']} to {results['maximum_pixel_value']}")
	print(f"NaN values: {results['nan_values']}; Inf values: {results['inf_values']}")
	print(f"Unreadable samples: {results['unreadable_sample_count']}")
	print(f"Validation JSON: {json_path}")
	print(f"Class distribution CSV: {csv_path}")
	print(f"Sample grid: {figure_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
