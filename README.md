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

Phase 1 uses the **MiraBest Zenodo v1 basic binary dataset** for the binary task. The validated archive contains 704 training samples and 88 test samples, for 792 total samples:

- label `0`: FR-I
- label `1`: FR-II
- FR-I samples: 351
- FR-II samples: 441

This Zenodo v1 archive is already a two-class FRI/FRII dataset. It must not be confused with the later full MiraBest archive, whose internal fine labels support the separate `MBFRConfident` definition of 833 confident samples. That later definition is not being used in Phase 1.

The official Zenodo v1 archive is expected at `data/raw/mirabest/batches.tar.gz` and is verified with MD5 `fa9d6ad8fdde95b0a19df11433e8bca3`. The loader preserves this archive and extracts the `batches/` directory only when it is not already present. Inspect and validate it with:

```text
python scripts/download_data.py
```

The command writes actual validation statistics to `outputs/metrics/dataset_validation.json` and `outputs/metrics/class_distribution.csv`, and saves a real sample grid to `outputs/figures/mirabest_sample_grid.png`. It does not train a model or preprocess/augment images.

## Usage

Create or activate a Python 3.12 environment, then install the pinned project dependencies with:

```text
pip install -r requirements.txt
```

Model implementation will be added after dataset validation and inspection are complete.
