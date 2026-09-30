# J26-IT-449 - AGN-Aware Radio Source Structure Classification

Final-year research project: **Explainable AI for Astronomical Discovery using XAI and Uncertainty Quantification**.

Individual component: **AGN-Aware Radio Source Structure Classification using CNN + XAI**.

Student: J.A.B.G. Jayawickrama  
Student ID: IT23308602

## Environment

- Python 3.12
- PyTorch

## Project Structure

- `data/`: Dataset storage and generated splits. Raw data is intentionally not included.
- `notebooks/`: Exploratory analysis, modelling, evaluation, and XAI workflow notebooks.
- `src/`: Reusable data, model, XAI, and utility modules.
- `scripts/`: Command-line entry points for the research pipeline.
- `outputs/`: Generated models, metrics, figures, and XAI artefacts.
- `report_assets/`: Figures and tables prepared for the final report.
- `tests/`: Automated tests for the project components.

## Dataset

Phase 1 uses the **MiraBest Zenodo v1 basic binary dataset** from the official [Zenodo record 10.5281/zenodo.4288837](https://zenodo.org/records/4288837). The checksum-verified local archive contains 704 training samples and 88 test samples, for 792 total images:

- label `0`: FR-I
- label `1`: FR-II
- FR-I samples: 351
- FR-II samples: 441

This Zenodo v1 archive is already a two-class FRI/FRII dataset. The Zenodo metadata page describes a 1256-image dataset, but direct auditing of the checksum-verified downloadable archive found only 792 raw images. This documented metadata/payload discrepancy is preserved rather than hidden. All experimental results in this implementation use the actually verified 792-image payload.

This payload must not be confused with the later full MiraBest fine-label archive and its separate `MBFRConfident` definition of 833 confident samples. The later definition is not used here.

The official archive is expected at `data/raw/mirabest/batches.tar.gz` and is verified with MD5 `fa9d6ad8fdde95b0a19df11433e8bca3`. The loader preserves this archive and extracts the `batches/` directory only when it is not already present. Inspect and validate it with:

```text
python scripts/download_data.py
```

The command writes actual validation statistics to `outputs/metrics/dataset_validation.json` and `outputs/metrics/class_distribution.csv`, and saves a real sample grid to `outputs/figures/mirabest_sample_grid.png`. It does not train a model or preprocess/augment images.

## Phase 2: Preprocessing and Splitting

Create normalized float tensors and reproducible stratified splits with:

```text
python -m src.data.split_dataset --config config.yaml
```

The current seed-42 split contains 554 training samples, 119 validation samples, and 119 test samples. Split indices and summaries are saved under `data/splits/`; the class-distribution figure is saved to `outputs/figures/split_class_distribution.png`. Images remain `150x150`, validation/test transforms are deterministic, and conservative rotations/flips are applied only through the training view. No CNN training is performed in Phase 2.

## Usage

Create or activate a Python 3.12 environment, then install the pinned project dependencies with:

```text
pip install -r requirements.txt
```

Further model architectures will be considered only after the baseline evaluation is complete.

## Phase 3: Baseline CNN

Start Model 1 baseline training with:

```text
python scripts/train_model.py --config config.yaml
```

The pipeline uses the existing seed-42 train/validation split, CrossEntropyLoss, Adam, train-only augmentation, and early stopping on validation loss. It saves only the best checkpoint to `outputs/models/baseline_cnn_best.pth`, along with training history, curves, and validation metrics under `outputs/metrics/` and `outputs/figures/`. The test split is intentionally not evaluated in Phase 3.

Model 2 is a moderately stronger CNN with three convolutional blocks using channels `32 -> 64 -> 128`, BatchNorm, ReLU, conservative max pooling, dropout, and a retained `4x4` spatial representation before classification. It remains CPU-compatible and does not use ResNet or Grad-CAM.

Start Model 2 training with:

```text
python scripts/train_model.py --model improved --config config.yaml
```

Model 2 writes only to its separate artifacts: `outputs/models/improved_cnn_best.pth`, `outputs/metrics/improved_training_history.csv`, `outputs/metrics/improved_validation_metrics.json`, and `outputs/figures/improved_training_curves.png`. The test split is not evaluated.

Model 3 is a lightweight ResNet-style CNN with three residual blocks, BatchNorm, ReLU, projection skip connections where channel dimensions change, and a retained `2x2` spatial representation. It has 339,746 trainable parameters and uses CPU-compatible Adam settings with learning rate `0.0003`, weight decay `0.0001`, and `ReduceLROnPlateau` scheduling.

Start Model 3 training with:

```text
python scripts/train_model.py --model resnet --config config.yaml
```

Model 3 writes only to `outputs/models/resnet_cnn_best.pth`, `outputs/metrics/resnet_training_history.csv`, `outputs/metrics/resnet_validation_metrics.json`, and `outputs/figures/resnet_training_curves.png`. The test split remains unevaluated.

## Viva Demo

The local Gradio prototype uses only the selected ResNet checkpoint and does not retrain the model. If Gradio is not installed, run:

```text
pip install gradio
```

Launch the demo with:

```text
python demo/app.py
```
