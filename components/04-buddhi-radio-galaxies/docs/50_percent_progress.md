# 50% Progress Milestone

## Project

**AGN-Aware Radio Source Structure Classification using CNN + XAI**  
**Student:** J.A.B.G. Jayawickrama  
**Student ID:** IT23308602

## 1. Project Objective

The completed machine-learning phase addresses binary classification of radio-galaxy morphology into Fanaroff-Riley class I (FR-I) and class II (FR-II).

The classification model is the foundation for the next research phase. Grad-CAM and other explainable-AI analyses are future work and are not part of this completed milestone.

## 2. Dataset Preparation

The implementation uses the MiraBest Zenodo v1 basic binary dataset. Direct auditing of the checksum-verified archive established 792 usable samples:

- FR-I: 351
- FR-II: 441
- Image representation: `1 x 150 x 150` grayscale

A fixed stratified split was created with seed 42:

| Split | Samples |
|---|---:|
| Train | 554 |
| Validation | 119 |
| Test | 119 |

Images are converted to float tensors and normalized using mean `0.5` and standard deviation `0.5`. The original image dimensions are preserved. Training-only augmentation consists of random rotations up to ±15° and horizontal/vertical flips. Validation and test data use deterministic preprocessing without augmentation.

## 3. Models Implemented

Three progressively stronger classifiers were implemented and compared using the same dataset, split, preprocessing, and validation protocol:

- Baseline CNN
- Improved CNN with BatchNorm and retained spatial features
- Lightweight ResNet-style CNN with residual blocks, BatchNorm, skip connections, and retained spatial features

### Validation Results

| Model | Validation Accuracy | Macro F1 | Balanced Accuracy | Validation Loss |
|---|---:|---:|---:|---:|
| Baseline CNN | 57.14% | 51.68% | 54.12% | 0.6794 |
| Improved CNN | 83.19% | 82.84% | 82.62% | 0.4500 |
| Selected ResNet checkpoint | 85.71% | 85.30% | 84.89% | 0.3501 |

The ResNet architecture was selected using validation performance before any test-set evaluation. The test set was not used for model selection.

## 4. Final Test Evaluation

After model selection, the locked ResNet checkpoint was evaluated once on the saved test split. The test set was not used to change the architecture, hyperparameters, or checkpoint.

| Metric | Result |
|---|---:|
| Test Accuracy | 83.19% |
| Macro F1 | 82.99% |
| Balanced Accuracy | 82.99% |
| Test Loss | 0.3464 |
| FR-I F1 | 81.13% |
| FR-II F1 | 84.85% |
| Correct predictions | 99 |
| Incorrect predictions | 20 |

The confusion matrix, with rows representing true labels and columns representing predicted labels, was:

```text
[[43, 10],
 [10, 56]]
```

This final test evaluation was performed only after the ResNet model had been selected using validation data.

## 5. Reproducibility Analysis

The same ResNet architecture was trained using the fixed train/validation split and the following random seeds:

```text
42, 7, 99, 123, 2026
```

Observed validation accuracies were:

| Seed | Validation Accuracy |
|---:|---:|
| 42 | 88.24% |
| 7 | 83.19% |
| 99 | 56.30% |
| 123 | 87.39% |
| 2026 | 83.19% |

Deterministic seeding was implemented for Python, NumPy, and PyTorch, including deterministic PyTorch settings where applicable. The same saved split indices were reused for every run.

Seed 99 produced a reproducible poor optimization trajectory. Its training loss and training accuracy improved while validation loss became unstable and validation accuracy remained near the majority-class level. This indicates sensitivity to some initializations and stochastic training dynamics. The result has not been hidden or excluded, and the experiments do not support a claim of perfect stability.

The two ResNet validation accuracy values have different experimental origins. The `85.71%` value belongs to the original model-selection checkpoint. The `88.24%` value belongs to the later deterministic seed-42 reproducibility and ablation run. The original locked checkpoint and final test result were not retroactively changed.

Aggregate reproducibility results across the five seeds were:

- Mean validation accuracy: 79.66%
- Standard deviation of validation accuracy: 11.87 percentage points
- Mean Macro F1: 75.97%

## 6. Ablation Study

A controlled seed-42 validation-only ablation study was performed with the same ResNet architecture, data split, preprocessing, batch size, learning rate, early stopping configuration, and class mapping. Each configuration changed only its named component.

| Configuration | Accuracy | Macro F1 | Balanced Accuracy | Validation Loss |
|---|---:|---:|---:|---:|
| Full ResNet | 88.24% | 87.99% | 87.72% | 0.3130 |
| No augmentation | 80.67% | 80.22% | 79.97% | 0.4005 |
| No weight decay | 88.24% | 87.86% | 87.35% | 0.3275 |
| No scheduler | 63.87% | 56.12% | 59.99% | 0.6522 |

The scheduler had the largest observed effect in this controlled comparison. Augmentation was beneficial. Weight decay had a smaller observed effect in this experiment. These observations are descriptive and should not be generalized beyond the tested configuration and seed without further study.

## 7. Error Analysis

Descriptive post-hoc error analysis was performed on the locked ResNet test predictions:

- Test samples: 119
- Correct predictions: 99
- Incorrect predictions: 20
- FR-I to FR-II errors: 10
- FR-II to FR-I errors: 10
- Mean confidence for correct predictions: 85.35%
- Mean confidence for incorrect predictions: 67.39%
- High-confidence errors with confidence >= 0.80: 3
- Low-confidence predictions with confidence < 0.60: 14

The high-confidence error samples had dataset indices:

```text
107, 497, 583
```

This was descriptive post-hoc analysis only. It was not used to tune the model, change hyperparameters, select a different checkpoint, or modify the final test result.

## 8. Demo

A simple local Gradio upload demo exists for the viva milestone. It:

- Loads the locked ResNet checkpoint
- Accepts one radio-galaxy image
- Predicts FR-I or FR-II
- Displays class probabilities and prediction confidence
- Performs inference only, without retraining or fine-tuning

This is a temporary individual viva/demo interface. The final group UI may be integrated later if required.

## 9. Current Completed Milestone

The completed ML classification milestone includes:

- MiraBest dataset validation
- Deterministic preprocessing and fixed stratified splitting
- Baseline CNN implementation
- Improved CNN implementation
- Lightweight ResNet-style CNN implementation
- Validation-based model comparison
- Final ResNet test evaluation after model selection
- Multi-seed reproducibility analysis
- Controlled validation-only ablation study
- Descriptive test-set error analysis
- Local inference demo

Grad-CAM and other XAI implementation are not complete and are not claimed as part of this milestone.

## 10. Next Phase

The next research phase includes:

- Grad-CAM implementation
- Explanation visualization
- Correct-versus-incorrect explanation analysis
- Explanation faithfulness evaluation
- Explanation stability evaluation
- Morphology and physics alignment analysis
- Final group UI integration if required
