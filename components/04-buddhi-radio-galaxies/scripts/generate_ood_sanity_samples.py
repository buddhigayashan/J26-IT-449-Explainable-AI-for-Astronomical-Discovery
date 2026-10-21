"""Generate deterministic synthetic images for OOD robustness sanity checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


OUTPUT_DIR = Path("data/ood/sanity")
SIZE = 150


def save_image(array: np.ndarray, name: str) -> Path:
	path = OUTPUT_DIR / name
	path.parent.mkdir(parents=True, exist_ok=True)
	Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="L").save(path)
	return path


def main() -> int:
	axis = np.linspace(0, 255, SIZE, dtype=np.float32)
	gradient = np.tile(axis, (SIZE, 1))
	checkerboard = ((np.indices((SIZE, SIZE)).sum(axis=0) % 2) * 255).astype(np.float32)
	rng = np.random.default_rng(42)
	images = {
		"black.png": np.zeros((SIZE, SIZE), dtype=np.float32),
		"white.png": np.full((SIZE, SIZE), 255, dtype=np.float32),
		"uniform_gray.png": np.full((SIZE, SIZE), 128, dtype=np.float32),
		"gaussian_noise.png": rng.normal(128, 35, (SIZE, SIZE)),
		"random_noise.png": rng.uniform(0, 255, (SIZE, SIZE)),
		"gradient.png": gradient,
		"checkerboard.png": checkerboard,
	}
	for name, array in images.items():
		print(save_image(array, name))
	return 0


if __name__ == "__main__":
	raise SystemExit(main())