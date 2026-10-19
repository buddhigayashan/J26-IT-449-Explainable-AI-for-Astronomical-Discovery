# Research Execution Evidence Registry

## 1. Purpose and scope

Project: **J26-IT-449-IT23308602-AGN-XAI**, AGN-Aware Radio Source Structure Classification using CNN and Explainable AI (FR-I vs FR-II).

Review date: **2026-10-19, Asia/Colombo** (client date; not an experiment timestamp). This Step 12B registry reconciles existing evidence only. The permitted path already contained an untracked draft; this document revises that draft. All other uncommitted work is preserved.

No training, inference, XAI experiment, calibration, pytest run, image decoding or TEST evaluation was performed for this registry. Inspection read source/text/JSON/CSV and hashed opaque bytes; the archive was not decoded and checkpoints were not deserialized. No receipt or historical metadata was created or altered.

## 2. Evidence classification rules

- **Verified now:** current byte hashes, saved values and metadata relations checked during this documentation task.
- **Saved execution evidence:** artifacts, process argv, approvals and reports supporting their recorded scope. They do not prove unrecorded executions, operator identity or scientific validity.
- **Implementation evidence:** source and synthetic fixtures describe intended behavior; presence is not proof of execution or passing tests.
- **Reported event:** supplied by the user or inherited draft without a standalone saved execution record. Do not promote it to independently verified execution.
- **Pending:** no completed corresponding real-data result found in inspected locations, not proof that no unrecorded execution ever occurred.

Use **Unknown** or **Not recorded** when evidence is absent. Mutable filesystem modification times are not authenticated execution dates. Git dates describe commits, not experiments. Hashes establish byte consistency, not authorship. Pixels, seeds, transformations and diagnostic calls are not independent image observations.

## 3. Repository and frozen research identifiers

Expected/observed branch: `feature/external-radio-input`. Expected short HEAD: `beb6974`; observed full HEAD: `beb69742d26dcbf546ec2c3e855bb0804f77ab3f`. No discrepancy. Commit subject: `feat(data): add bounded MiraBest archive extraction`; author/committer date `2026-10-19T18:26:01+05:30` is Git metadata, not an execution date. No Git write operation occurred.

Initial status contains modified tracked source/tests/docs and untracked configuration, XAI implementation/tests/evidence and this registry. Their presence is not approval to stage them together. Git reports `.pytest_cache/` Permission denied; status visibility for that directory is limited. This is not evidence of dataset corruption.

The selected lightweight [ResNetCNN](../src/models/resnet_cnn.py) has a one-channel stem, residual widths 32/64/128, adaptive 2x2 pooling and Linear 512->64->2. FR-I=0, FR-II=1. Grad-CAM target: `model.blocks[2].convolution[3]`, native 19x19, output 150x150.

Frozen provenance identifier **F** below means these current identities, verified by byte hashing:

| Reference | Verified SHA-256 |
|---|---|
| [outputs/models/resnet_cnn_best.pth](../outputs/models/resnet_cnn_best.pth) | `d5e1f126c13c2b7bebb95e35f41f0bc6397dd0ea9f4fbfcfbc55648609ab5bc7` |
| [data/splits/split_indices.json](../data/splits/split_indices.json) | `c4a548a66dd37505b8d663bfe6826258364b8118afde4928f5f83a6cdd263008` |
| [data/raw/mirabest/batches.tar.gz](../data/raw/mirabest/batches.tar.gz) | `505e0b0761a76364a417695380d32d9f26cca7bc0a2608cad46ab338fa46a6b8` |
| [config.yaml](../config.yaml) | `2660615806a50b5f634ff4681a8e560398402559b42478ad7eaafcdcc9694e6e` |
| [config/xai_phase3b_v1.0.0.yaml](../config/xai_phase3b_v1.0.0.yaml) | `9817bd245acf6a851a9eb0c161132512c0880fa2599e0ff82a0c8502fc0e474e` |
| [docs/xai/phase3b_protocol_v1.0.0.md](../docs/xai/phase3b_protocol_v1.0.0.md) | `319583cc445f80dc5f60bd5ac33fb4e65197aad9a5d319de509b105ac548964a` |


Archive MD5: `fa9d6ad8fdde95b0a19df11433e8bca3` (verified). Protocol: **Phase3B-XAI-v1.0.0**. Canonical parsed-YAML SHA-256: `45fb3e86b681c6f73035a7743fa0223eb271445ee6ea06bf9a2604551ef3ad84` (verified independently from raw-file hashing). Recorded metadata SHA-256: `0d8a164b7e68ae93d152a36a6c90125f80ffa6bdf23327d0b9160d52f042b5b7`; not independently regenerated from archive metadata here.

All **41 protected artifact hashes** match both XAI preflight maps: files under `outputs/models`, `outputs/metrics`, `outputs/ood` and `data/splits`. This map does not cover every output directory. F does not retroactively bind historical training/TEST events that did not save hashes. Current HEAD is not established as the historical XAI execution commit; source hashes bind worktree bytes instead.

Both XAI runs' recorded implementation hashes match current files, including dirty/untracked files:

| Source | Verified SHA-256 |
|---|---|
| [src/data/preprocess.py](../src/data/preprocess.py) | `e84921262055576282966b908f9988d4fa63dd576efe2d217df07e258a4a088d` |
| [src/xai/evaluation_data.py](../src/xai/evaluation_data.py) | `1e2f3bcf9ec6ea7e6b8e5ab09cfea8c86ce77cbbfce89a47de0c76f82e5be6a7` |
| [src/xai/faithfulness.py](../src/xai/faithfulness.py) | `e2bba7f0baeed369904029ec15ed174209ab6d4f367f90bf80b53dd50b71a69a` |
| [src/xai/gradcam.py](../src/xai/gradcam.py) | `cf97ee36b3f7e46ea92d71351b5022fd89efaef169015cfe85ec40bcc25cbbde` |
| [src/xai/morphology_audit.py](../src/xai/morphology_audit.py) | `e3b71cc9056f436344414f5df12c6f680bc726a8368bae12ae2b03a6dd832157` |
| [src/xai/sanity_checks.py](../src/xai/sanity_checks.py) | `c97f22244bf49d839addea9aa0a27b1368f2027bde7699a015b78264357ba622` |
| [src/xai/stability.py](../src/xai/stability.py) | `628c82d35eafd7ed79a80b329739123ac09aed5d46846f29e9e6a42eb58296ae` |
| [src/xai/statistics.py](../src/xai/statistics.py) | `0b28ad291e5bedeb50dce3f374731e29fb2b427a24fa1fda4d28f430033b5b66` |
| [scripts/run_xai_validation_evaluation.py](../scripts/run_xai_validation_evaluation.py) | `d2a0553718f71734430b4048b4ce3022ca385932b4380e7ec3b94aad21f849d6` |
| [src/xai/execution_adapter.py](../src/xai/execution_adapter.py) | `cc9c7011991ba0919df935057ee8cc2d6b42837342b10de3e9dddde30aad20a4` |
| [src/models/resnet_cnn.py](../src/models/resnet_cnn.py) | `a6b2fa85f2c7ae8c9ae7db853556b21626e78583ecc36b6e4f069692d67cf187` |


## 4. Dataset validation and integrity events

**DATA-01 - historical dataset validation.** Type/purpose: dataset inspection. [dataset_validation.json](../outputs/metrics/dataset_validation.json) reports 792/792 valid MiraBest records, 351 FR-I and 441 FR-II, 150x150 uint8, no NaN/Inf, unreadable records or invalid labels; preprocessing not applied. Source field cites a GitHub loader URL while [config.yaml](../config.yaml) cites Zenodo record 4288837. A loader URL is not a download receipt. Dataset/split: whole local payload; checkpoint not applicable. Exact command, approval, trusted date, historical archive/source hashes: **Not recorded**. Status: saved report exists; its image checks were not rerun.

**DATA-02 - documentation-time integrity reconciliation.** Verified F archive SHA-256/MD5, checkpoint, split, protocol and both protected maps. Parsed saved split lists contain 554 TRAIN, 119 VALIDATION and 119 TEST indices: disjoint, unique and covering 0..791. Date: registry review date only. Approval: Step 12B request. Command: ad hoc read-only inspection, no standalone command log created. Output: this registry. No image validity, duplicate-free status or source independence is established by these checks. Archive batch partitions are not research splits; use frozen global indices for TEST exclusion.

**OOD-01 - saved frozen research calibration.** [ood_calibration.json](../outputs/ood/ood_calibration.json), paired [PT artifact](../outputs/ood/ood_calibration.pt), [script](../scripts/calibrate_ood_detector.py). JSON records F checkpoint/split hashes; TRAIN 554 for feature statistics and VALIDATION 119 for thresholds, with saved index lists matching the split and excluding TEST. Energy threshold `-0.49593809247016907`, feature-distance threshold `70.37763214111328`, percentile `0.99`; 512 pooled features; grayscale 150x150 PIL-bilinear preprocessing, normalization mean/std 0.5, no augmentation. PT bytes match the protected maps, but were not loaded. Exact command, separate approval, reliable date, execution commit: **Not recorded**. Status: calibration artifacts present, not recalibrated. Source restrictions do not independently prove every historical loader operation.

**OOD-02 - external synthetic compatibility calibration/development.** [external_compatibility_calibration.json](../outputs/ood/external_compatibility_calibration.json) and paired PT record F checkpoint/split, seed 42, transformations/variant counts and synthetic positive-only calibration. This is distinct from OOD-01, with different thresholds; do not substitute it into research diagnostics. [pseudocolor_development_checks.json](../outputs/ood/pseudocolor_development_checks.json) records 3C31 and Cygnus A development examples, both `UNSUPPORTED`, confidence null, probabilities empty. Their external paths/hashes were not opened or independently verified here. They are expressly ineligible for untouched final external evaluation. Exact command, approval, reliable dates and historical source manifest: **Not recorded**. Status: saved calibration/development evidence, not natural-image false-accept measurement or independent external-survey validation.

## 5. CNN training and validation events

Shared qualification for CNN-01 through CNN-09: exact commands, original approvals, authenticated dates, execution commits and complete per-event hash/environment manifests are **Not recorded** in inspected results. Current relevant source: [train.py](../src/models/train.py), [evaluate.py](../src/models/evaluate.py), [config.yaml](../config.yaml). Training API accepts TRAIN/VALIDATION only, uses Adam and minimum validation-loss selection. Current source describes behavior, not an unlogged historical invocation. Saved `test_evaluation_performed: false` is an artifact assertion, not independent access telemetry. F is the current archive/split reference; historical byte binding is Unknown unless recorded. Status: saved result/history artifacts present, no new execution or claim about total attempt count. Validation informed development and is not independent external validation.

| Event | Type/scope | Saved results | Artifacts | Checkpoint |
|---|---|---|---|---|
| CNN-01 | baseline; current saved scope 554 TRAIN / 119 VALIDATION | Best epoch 20; completed 20; loss 0.679418729133; accuracy 0.571428571429; macro-F1 0.516837831383 | [outputs/metrics/baseline_validation_metrics.json](../outputs/metrics/baseline_validation_metrics.json); [outputs/metrics/baseline_validation_evaluation.json](../outputs/metrics/baseline_validation_evaluation.json); [outputs/metrics/training_history.csv](../outputs/metrics/training_history.csv) | [outputs/models/baseline_cnn_best.pth](../outputs/models/baseline_cnn_best.pth); current protected bytes verified; event-time byte binding Unknown |
| CNN-02 | improved; current saved scope 554 TRAIN / 119 VALIDATION | Best epoch 10; completed 17; loss 0.449978639599; accuracy 0.831932773109; macro-F1 0.828431372549 | [outputs/metrics/improved_validation_metrics.json](../outputs/metrics/improved_validation_metrics.json); [outputs/metrics/improved_validation_evaluation.json](../outputs/metrics/improved_validation_evaluation.json); [outputs/metrics/improved_training_history.csv](../outputs/metrics/improved_training_history.csv) | [outputs/models/improved_cnn_best.pth](../outputs/models/improved_cnn_best.pth); current protected bytes verified; event-time byte binding Unknown |
| CNN-03 | resnet; current saved scope 554 TRAIN / 119 VALIDATION | Best epoch 22; completed 29; loss 0.350147131110; accuracy 0.857142857143; macro-F1 0.852990335005 | [outputs/metrics/resnet_validation_metrics.json](../outputs/metrics/resnet_validation_metrics.json); [outputs/metrics/resnet_validation_evaluation.json](../outputs/metrics/resnet_validation_evaluation.json); [outputs/metrics/resnet_training_history.csv](../outputs/metrics/resnet_training_history.csv) | [outputs/models/resnet_cnn_best.pth](../outputs/models/resnet_cnn_best.pth); current protected bytes verified; event-time byte binding Unknown |


CNN-03 selects epoch 22 by minimum validation loss even though epoch 27 has higher validation accuracy. This matches the selection rule. The seed-42 reproducibility/full-ablation result differs from the main selected run; it must not replace that run's metrics.

**CNN-04 - five-seed reproducibility.** [summary](../outputs/reproducibility/resnet_multiseed_summary.json), [results](../outputs/reproducibility/resnet_multiseed_results.csv), per-seed metrics/history and [script](../scripts/run_multiseed_resnet.py). Seeds 42, 7, 99, 123, 2026; TRAIN/VALIDATION only according to saved flags. Mean validation accuracy `0.7966386554621849`, standard deviation `0.11865117710738012`; mean macro-F1 `0.7596832357378731`. Checkpoints under `outputs/models/reproducibility/` are current protected files. Historical event-specific command/hash/date/approval binding: Not recorded. Five saved seed results do not prove only five attempted runs.

**CNN-05/06/07 - verification artifacts.** `outputs/reproducibility_seed42_verify1/reproducibility/`, `outputs/reproducibility_seed42_verify2/reproducibility/` and `outputs/reproducibility_seed99_verify/reproducibility/` each hold one-seed summaries, metrics and histories with no-TEST flags. Seed-42 validation accuracy is `0.8823529411764706` in both directories; seed-99 accuracy `0.5630252100840336`. Matching metrics alone do not prove identical tensors, independent executions or comprehensive reproducibility. Historical commands, approvals, dates and checkpoint identities: Not recorded.

**CNN-08 - validation ablations.** [results](../outputs/ablation/ablation_results.csv), per-configuration JSON/history under `outputs/ablation/`, [script](../scripts/run_ablation_study.py), checkpoints under `outputs/models/ablation/`. Seed 42; `full_resnet`, `no_augmentation`, `no_scheduler`, `no_weight_decay`; saved validation accuracies respectively `0.8823529411764706`, `0.8067226890756303`, `0.6386554621848739`, `0.8823529411764706`. Saved flags exclude TEST. Exact commands, approvals, reliable dates and historical checkpoint/config hash binding: Not recorded. These are development comparisons, not untouched validation.

**CNN-09 - rotation comparison.** [results](../outputs/augmentation_experiment/rotation_results.csv), [experiment_config.json](../outputs/augmentation_experiment/experiment_config.json), paired histories/metrics, [script](../scripts/run_rotation_augmentation_experiment.py); checkpoints under `outputs/models/augmentation_experiment/`. Saved train/validation lists match F. Seed 42, ResNetCNN, Adam, batch 32, LR 0.0003, weight decay 0.0001, scheduler/early stopping, mean/std 0.5. Random +/-15-degree versus 0/90/180/270-degree rotations, both with horizontal/vertical flips. Best epochs 21/11; losses `0.3130206683603655`/`0.44315003747699644`; accuracies `0.8823529411764706`/`0.8487394957983193`; macro-F1 `0.8799019607843137`/`0.8462973593570609`. Saved flags exclude TEST. Original commands, approvals, dates and event-time checkpoint hashes: Not recorded. Separate experiment artifacts do not imply modification of the locked checkpoint.

## 6. Historical TEST evaluation

**TEST-01 - historical classification result.** Evidence: [resnet_test_evaluation.json](../outputs/metrics/resnet_test_evaluation.json), [final_test_metrics.csv](../outputs/metrics/final_test_metrics.csv), [evaluate_test.py](../scripts/evaluate_test.py), [evaluate.py](../src/models/evaluate.py). Saved scope: TEST, 119 samples. Recorded checkpoint path ends in `outputs/models/resnet_cnn_best.pth`; TEST JSON does not record its hash. Current identity is F; **original checkpoint bytes at evaluation time are Unknown**. Exact per-event indices/hash are not embedded; current script uses the frozen TEST list.

Verified shared JSON/CSV values: accuracy **0.8319327731092437**, macro-F1 **0.8299028016009149**, balanced accuracy **0.8299028016009149**, loss **0.34639566986500714**. JSON confusion matrix **[[43,10],[10,56]]**, 119 total, 99 correct and 20 incorrect. Related [error summary](../outputs/error_analysis/error_summary.json) exists but a direct authenticated event link is not established.

Status: saved historical metrics exist. Exact command, approval receipt, execution count, trusted start/end dates, source commit and historical dataset/split/checkpoint hashes: **Unknown / Not recorded**. The script's 'exactly once' wording is intent, not an execution counter. Its current eager loader bulk-loads before subset selection; that code is not an independent trace of the historical invocation. Do not claim TEST ran exactly once. No TEST was rerun here.

**TEST-OOD-01 - separate saved descriptive TEST OOD evaluation.** [ood_evaluation.json](../outputs/ood/ood_evaluation.json), [ood_predictions.csv](../outputs/ood/ood_predictions.csv), [script](../scripts/run_ood_evaluation.py). Saved 119 ID TEST samples: 118 accepted, 1 rejected; 7 OOD samples: 7 rejected. Paths identify F checkpoint and OOD-01 calibration. Recorded energy AUROC `0.18247298919567825`, FPR95 `1.0`; distance AUROC `1.0`, FPR95 `0.0`; higher scores more OOD-like. Threshold provenance states TRAIN statistics/VALIDATION thresholds; do not claim fitting on TEST. Status: saved results, not rerun. Exact command, approval, trusted date, event-time checkpoint/calibration bytes and external OOD selection history: Not recorded. Seven OOD samples cannot establish general external validity. This additional TEST-related artifact must not be concealed by a claim of exactly one TEST use.

## 7. Bounded MiraBest reader pilot

**READER-01 - reported eight-image real engineering pilot.** Purpose: selected TRAIN/VALIDATION extraction excluding TEST materialization. Step 12B's user statement reports successful extraction of eight images. The inherited untracked draft reports TRAIN `[0,195,399,592]`, VALIDATION `[31,176,393,574]`, 180,000 bytes, no selected-subset exact duplicates, and zero forbidden materializations. These detailed measurements are **reported, not independently corroborated by a standalone saved pilot report located here**. Listed indices agree with frozen memberships. No CNN/checkpoint required; current archive/split pins are F, per-event pin verification not independently preserved.

Relevant [production reader](../src/data/mirabest_pickle_stream_reader.py) scans a restricted inert protocol-3 grammar, does not execute pickle, and limits explicit extraction to eight images. Current SHA-256 `981da0039a69d0e743093cd3122a07b0b873f0984274f21d19656fc7ef06a9f8`. [Synthetic tests](../tests/test_mirabest_pickle_stream_reader.py) assert bounded extraction and forbidden-row behavior; these fixture assertions are not a real pilot log. The draft's reported 35 focused passing tests has no standalone test-run log found here; no tests were rerun.

Exact pilot command, repository approval receipt, authenticated date, execution-time source hash and standalone instrumentation: **Not recorded / not found**. Status: reported successful bounded pilot with incomplete provenance.

**DUP-01 - full real duplicate audit pending.** All **673 TRAIN+VALIDATION** records would be required, not merely the eight selected images. [run_duplicate_audit_train_val.py](../scripts/run_duplicate_audit_train_val.py) currently imports the isolated `MBRAW1` synthetic reader and explicitly refuses real MiraBest execution in `main`; the later bounded reader alone does not wire a completed full audit. Current script SHA-256 `fccb0ed2e67dd72fb5e0eb1828af6ecb8a324d0506d1c9c3c08e121004e33f16`. [Synthetic duplicate tests](../tests/test_duplicate_audit_train_val.py) are implementation evidence. No completed 673-image report, command or execution approval was found. Exact-image duplicates and astronomical source-level independence are different questions; coordinate-like filenames are not verified source IDs.

## 8. XAI exploratory execution - two-image pilot

**XAI-3B-P2** - bounded real VALIDATION engineering/exploratory execution, not final scientific validation.

| Field | Evidence-supported value |
|---|---|
| Purpose | Verify integrated metric execution and operational safeguards on a bounded cohort. |
| Dataset/scope | F MiraBest VALIDATION; indices `[31, 50]`, `max_images=2`. No TRAIN/TEST authorized. |
| Code/config | Runner, adapter and module hashes in section 3; frozen Phase3B-XAI-v1.0.0 YAML/document; trusted deterministic preprocessing. |
| Checkpoint/archive/split | F; image provenance embeds these hashes. Current file bytes match. |
| Approval | [outputs/xai/phase3b_pilot_31_50/approval.json](../outputs/xai/phase3b_pilot_31_50/approval.json); approved=true; SHA-256 `71be41b0d1006307bb80fce7a67357aeb4e1f09ff59f9287beaac2afdf1d4811`. |
| Approval context | `15f2119921f1e80dba24aeeda588397b49e465a4056dca6abcae553b80cb457d`; exact indices/count, CPU, Torch threads=1, preprocessing and class mapping. Local operator gate, not signed authentication. |
| Outputs | [outputs/xai/phase3b_v1.0.0/real_validation/323bb0f19f5b8768/run.json](../outputs/xai/phase3b_v1.0.0/real_validation/323bb0f19f5b8768/run.json); `image_NNNN.json`; [outputs/xai/phase3b_v1.0.0/real_validation/323bb0f19f5b8768/summary.json](../outputs/xai/phase3b_v1.0.0/real_validation/323bb0f19f5b8768/summary.json); [outputs/xai/phase3b_pilot_31_50/integrity_report.json](../outputs/xai/phase3b_pilot_31_50/integrity_report.json); stdout/status/preflight in same operational directory. |
| Status | Return code 0; 2 complete images; original CAMs valid, 19x19 native and 150x150 output. Implemented families executed; unavailable observations retained. complete_frozen_protocol_coverage=false. |
| Duration/resources | Whole process 334.544005s; per-image 150.044949-160.784839s. Process peak 344.18 MiB. Cooperative 2 GiB/900s per-image limits. |
| Date provenance | Authenticated start/end: Not recorded. Status file last-write observed 2026-10-18T21:05:13.984062+05:30 (Asia/Colombo), mutable filesystem metadata only. Elapsed durations are recorded. |
| Scientific limits/missing information | Small engineering cohort; no scientific performance claims. Four approved unavailable analyses remain incomplete. No authenticated operator identity, execution commit or signed timestamp chain. |

Exact recorded process argv from [outputs/xai/phase3b_pilot_31_50/execution_status.json](../outputs/xai/phase3b_pilot_31_50/execution_status.json) (JSON array, not a reconstructed shell-history claim):

```json
[
  "C:\\Users\\User\\Desktop\\research\\J26-IT-449-IT23308602-AGN-XAI\\.venv\\Scripts\\python.exe",
  "scripts/run_xai_validation_evaluation.py",
  "--mode",
  "real",
  "--project-root",
  "C:\\Users\\User\\Desktop\\research\\J26-IT-449-IT23308602-AGN-XAI",
  "--index",
  "31",
  "--index",
  "50",
  "--max-images",
  "2",
  "--max-working-set-bytes",
  "2147483648",
  "--max-image-seconds",
  "900",
  "--approval-file",
  "C:\\Users\\User\\Desktop\\research\\J26-IT-449-IT23308602-AGN-XAI\\outputs\\xai\\phase3b_pilot_31_50\\approval.json"
]
```

| Index | Ground truth | Predicted | Confidence | CAM |
|---|---|---|---|---|
| 31 | FR-I | FR-I | 0.695405217094 | valid |
| 50 | FR-II | FR-II | 0.997457317850 | valid |

## 9. XAI exploratory execution - eight-image batch

**XAI-3B-B8** - bounded real VALIDATION engineering/exploratory execution, not final scientific validation.

| Field | Evidence-supported value |
|---|---|
| Purpose | Verify integrated metric execution and operational safeguards on a bounded cohort. |
| Dataset/scope | F MiraBest VALIDATION; indices `[35, 37, 38, 51, 52, 53, 55, 94]`, `max_images=8`. No TRAIN/TEST authorized. |
| Code/config | Runner, adapter and module hashes in section 3; frozen Phase3B-XAI-v1.0.0 YAML/document; trusted deterministic preprocessing. |
| Checkpoint/archive/split | F; image provenance embeds these hashes. Current file bytes match. |
| Approval | [outputs/xai/phase3b_batch8/approval.json](../outputs/xai/phase3b_batch8/approval.json); approved=true; SHA-256 `1a946d0fe1c8a3f2e27a95f4dd03888b81a87936e89064a03813c58de46373b1`. |
| Approval context | `15f2119921f1e80dba24aeeda588397b49e465a4056dca6abcae553b80cb457d`; exact indices/count, CPU, Torch threads=1, preprocessing and class mapping. Local operator gate, not signed authentication. |
| Outputs | [outputs/xai/phase3b_v1.0.0/real_validation/73452038829db01e/run.json](../outputs/xai/phase3b_v1.0.0/real_validation/73452038829db01e/run.json); `image_NNNN.json`; [outputs/xai/phase3b_v1.0.0/real_validation/73452038829db01e/summary.json](../outputs/xai/phase3b_v1.0.0/real_validation/73452038829db01e/summary.json); [outputs/xai/phase3b_batch8/integrity_report.json](../outputs/xai/phase3b_batch8/integrity_report.json); stdout/status/preflight in same operational directory. |
| Status | Return code 0; 8 complete images; original CAMs valid, 19x19 native and 150x150 output. Implemented families executed; unavailable observations retained. complete_frozen_protocol_coverage=false. |
| Duration/resources | Whole process 393.398387s; per-image 45.232681-50.245635s. Process peak 356.07 MiB. Cooperative 2 GiB/900s per-image limits. |
| Date provenance | Authenticated start/end: Not recorded. Status file last-write observed 2026-10-19T09:33:15.141064+05:30 (Asia/Colombo), mutable filesystem metadata only. Elapsed durations are recorded. |
| Scientific limits/missing information | Small engineering cohort; no scientific performance claims. Four approved unavailable analyses remain incomplete. No authenticated operator identity, execution commit or signed timestamp chain. |

Exact recorded process argv from [outputs/xai/phase3b_batch8/execution_status.json](../outputs/xai/phase3b_batch8/execution_status.json) (JSON array, not a reconstructed shell-history claim):

```json
[
  "C:\\Users\\User\\Desktop\\research\\J26-IT-449-IT23308602-AGN-XAI\\.venv\\Scripts\\python.exe",
  "scripts/run_xai_validation_evaluation.py",
  "--mode",
  "real",
  "--project-root",
  "C:\\Users\\User\\Desktop\\research\\J26-IT-449-IT23308602-AGN-XAI",
  "--index",
  "35",
  "--index",
  "37",
  "--index",
  "38",
  "--index",
  "51",
  "--index",
  "52",
  "--index",
  "53",
  "--index",
  "55",
  "--index",
  "94",
  "--max-images",
  "8",
  "--max-working-set-bytes",
  "2147483648",
  "--max-image-seconds",
  "900",
  "--approval-file",
  "C:\\Users\\User\\Desktop\\research\\J26-IT-449-IT23308602-AGN-XAI\\outputs\\xai\\phase3b_batch8\\approval.json"
]
```

| Index | Ground truth | Predicted | Confidence | CAM |
|---|---|---|---|---|
| 35 | FR-I | FR-I | 0.946027151449 | valid |
| 37 | FR-I | FR-I | 0.866280158617 | valid |
| 38 | FR-I | FR-I | 0.675852288029 | valid |
| 51 | FR-II | FR-II | 0.916643903017 | valid |
| 52 | FR-II | FR-II | 0.810709074624 | valid |
| 53 | FR-II | FR-II | 0.890503035838 | valid |
| 55 | FR-II | FR-II | 0.964818272780 | valid |
| 94 | FR-I | FR-I | 0.522001031645 | valid |

**XAI-3A-01 - earlier visual verification, separate event.** [metadata.json](../outputs/xai/gradcam_phase3a/metadata.json), [generate_gradcam.py](../scripts/generate_gradcam.py), exported figures/arrays under `outputs/xai/gradcam_phase3a/`. Four VALIDATION indices `[31,35,50,51]`: two per class, lowest eligible saved indices in the original archive training partition according to its recorded rule, not heatmap-selected. F checkpoint/split hashes embedded. All predicted classes match ground truth, CAMs valid, baseline-logit equality flags true; target `blocks.2.convolution.3`. Inspected metadata only, not images/tensors. Exact command, standalone approval and authenticated date: Not recorded. Status: exploratory exports exist, not validated scientific explanations. Overlapping images are not extra unique Phase 3B observations.

## 10. Cross-run integrity findings

Both Phase 3B run manifests match scientific context, module hashes, schema, class mapping, preprocessing and seeded definitions. Only execution scopes differ. Recorded definitions use deletion seeds 1000-1019, noise seeds 2000-2009 at each sigma, sanity seeds 3000-3004, 10,000 bootstrap replicates with seed 42 and 10,000 sign-flip permutations with seed 43. These are fixed protocol controls, not independent image observations. Their saved integrity-report output hash maps match current run/image/summary bytes. Every canonical image envelope's payload and run digests match. The distinct union is `[31,35,37,38,50,51,52,53,55,94]`, five per ground-truth class, wholly VALIDATION and disjoint from TEST: **10/119** Phase 3B images, **109 remaining**, not complete scientific validation. No additional real-validation namespace was found in the inspected directory.

Saved integrity reports attest model/BatchNorm preservation and no locked TEST materialization; matching reviewed source authorizes before selected-row decoding. This supports isolation within that mechanism, not independent operating-system access logging or all older eager-loading pathways. No pixels were decoded for this registry.

These records can support a future compatibility-checked union keyed by dataset index. Do not average batch summaries or count overlapping Phase 3A exports again. Nothing was merged or rewritten here. Final analysis would require exactly 119 unique validation records; metric-specific available denominators may be smaller. Keep repeats within images.

## 11. Approval and authorization boundaries

Frozen YAML still records **execution_authorized: false**, stage 3B.1. This global declaration is not amended or fulfilled by bounded run receipts. Separate approvals bind `[31,50]` max 2 and `[35,37,38,51,52,53,55,94]` max 8 through the existing adapter gate. They authorize neither unrestricted/full-cohort execution nor future batches, TRAIN/TEST access or new conventions. Receipts are local exact-content gates, not cryptographically signed user identities.

Conversational approvals support those historical scopes; they must not be copied into new receipts. Four secondary controls were approved as unavailable for integration: an implementation limitation, not protocol amendment/completion. Older events lack repository approval receipts in inspected evidence; absence is a provenance gap, not proof of unauthorized execution. Step 12B authorizes this document only. No future execution approval is implied.

## 12. Known anomalies and scientific limitations

Evidence: per-image JSON in both real-validation namespaces, operational integrity reports, and matching runner/stability source. No causal follow-up was executed.

- **Index 94 original OOD rejection:** energy `-0.49353718757629395` exceeds frozen `-0.49593809247016907`; distance `12.174494743347168` is below `70.37763214111328`. Saved reason is energy failure. This is in `ood_diagnostics.calls` and the operational report, but absent from scalar `record.metrics` and summary original-input OOD acceptance coverage. 'Evaluated' does not mean 'OOD accepted'. Trusted-validation metrics include index 94 unfiltered; do not silently exclude it or change thresholds.
- **FR-II noise flips:** `50,51,52,53,55` predict FR-I under all 20 predefined conditions: sigma 0.005/0.01, seeds 2000-2009. Fixed attribution target remains original predicted FR-II. Saved probability/margin changes and classes are internally consistent in the prior read-only audit. This supports model-output changes rather than target/bookkeeping errors, but transformed images/logits were not saved for independent replay. Causes remain Unknown.
- **Clipping:** recorded noise clipped fractions span 47.36-50.23%. Gaussian additions are clipped to [0,1]; resulting fields are not unrestricted zero-mean Gaussian noise. These processed-intensity sensitivity tests do not establish physically calibrated flux robustness. Saved records cannot isolate clipping as the cause of flips.
- **Degenerate transformed CAMs:** `transformed_degenerate_cam` makes Jaccard unavailable for 51 noise comparisons: index 51 all 20; 52 all ten sigma=0.01; 53 sigma=0.01 seed 2001; 55 all 20. Predictions remain recorded when similarity is unavailable. At index 55, randomized stage 1 seeds 3002/3004 are unavailable with `candidate_degenerate`. Original CAMs are valid; unavailable explanations are not zero-valued successes.
- **Negative AOPC:** 272 probability, 280 logit and 279 margin negative values across 660 primary ranking/replacement curves are preserved. Image-level metrics and summaries also retain negative values. Repeated curves are not independent observations; positive aggregates need not imply every curve is positive.
- **Repeated diagnostics:** 408 rejected calls across ten images do not mean 408 images. The batch's 325 rejections include three repeated original checks for index 94. `classifier_curves_filtered=false`: rejected perturbations stay in deletion curves. Recorded diagnostic scope is deletion scoring, not evidence of OOD checks on every stability transformation.
- **Resources:** pilot mean 155.41s/image versus batch 47.12s/image; telemetry cannot establish CPU, thermal, power or contention causes. Peak process measurement 356.07 MiB does not establish a long-run memory plateau. Limits are cooperative; blocking operations are not immediately interruptible; sanity checking is at family boundaries.
- **Summary label defect:** real summaries retain `synthetic_storage_planning` and synthetic-only explanatory text. This is a reporting limitation; use measured file sizes for storage planning. Do not overwrite historical summaries without approval. Original-input OOD coverage also needs explicit future reporting.
- **Scientific scope:** validation informed development; ten selected engineering images do not establish final XAI performance. Emission masks are intensity proxies, not verified jets/lobes/cores/hotspots. Filename coordinate-like keys are not verified source IDs; source independence is unproven. Randomization retains learned BatchNorm buffers, so its final model is not wholly fresh. Sign-flip assumptions and availability-conditional intervals remain limitations; bootstrap uncertainty excludes retraining/external-survey variation.

Four approved unavailable items remain explicit, never treated as zero or completed:

| Analysis | Machine-readable reason |
|---|---|
| Secondary native 19x19 stability | `native_grid_pooling_partial_support_ties_unspecified` |
| Random emission-map baseline | `random_native_to_input_projection_unspecified` |
| Sanity brightness-map control | `sanity_brightness_definition_unspecified` |
| Sanity random-map control | `sanity_random_distribution_grid_seeds_replicates_projection_unspecified` |

## 13. Missing provenance and unresolved questions

Historical TEST run count/command, original evaluation checkpoint bytes, authenticated times and approvals are Unknown. Neither filenames nor one-time source comments resolve them. Training, reproducibility, ablation, augmentation, OOD and Phase 3A outputs lack complete historical command/environment/source/checkpoint chains. Current configuration cannot be substituted for missing immutable historical configuration.

The bounded-reader pilot lacks a standalone real execution report/command/receipt; detailed inherited draft claims are not independently verified. Test fixtures cannot fill that gap. XAI hashes match current sources but do not establish execution commit/operator identity. Metadata hash was not independently regenerated here. Mutable timestamps are not trusted execution times.

No full real duplicate audit, independently verified astronomical source grouping or untouched external-survey accuracy is established. Possible source leakage remains unresolved. Noise/clipping mechanisms, repeated-forward reproducibility, CAM degeneracy mechanisms and near-threshold numerical sensitivity require separately authorized experiments, not invented explanations.

## 14. Pending experiments and authorization requirements

1. **Full real duplicate audit: pending**, 673 TRAIN+VALIDATION images. Requires separately approved safe-reader integration and scope; no TEST pixels. Current synthetic-only entry point is not a completed real audit. Exact-image uniqueness does not establish source independence.
2. **Remaining validation evaluation: pending**, 109 images in the inspected evidence. Each future bounded batch needs separate explicit approval, fresh exact-scope receipt/namespace, protected/provenance checks, resource measurements, sequential failure-stop behavior and post-run integrity audit. No new receipt is created by this document.
3. **Final 119-image statistics/figures: not completed as a cohort.** Compatibility-check envelopes/source/provenance, reject duplicates, require exact allowlist coverage, retain paired values/repeats/missingness and frozen statistics. Do not combine batch p-values or silently discard OOD-rejected inputs. Future reporting corrections need separate authorization and must preserve historical evidence.
4. **Unspecified secondary metrics/mechanistic follow-up: unavailable or pending.** Require approved definitions before implementation; do not invent native-grid pooling, random projection or brightness-control conventions.

No new TEST, training, calibration or full validation is authorized here. If separately requested, the next safe Git commit is this reviewed registry alone, for example `docs: add research execution evidence registry`. Review XAI code/tests, external-radio changes and generated evidence separately; do not stage the whole dirty worktree. No staging, commit, push, stash, clean, reset or branch switch occurred in Step 12B.

## 15. Evidence file reference index

Pointers identify inspected evidence, not authenticated authorship, tracking status or scientific validity. Opaque archive/checkpoint links denote hashed files, not decoded content.

- [config.yaml](../config.yaml)
- [config/xai_phase3b_v1.0.0.yaml](../config/xai_phase3b_v1.0.0.yaml)
- [data/raw/mirabest/batches.tar.gz](../data/raw/mirabest/batches.tar.gz)
- [data/splits/split_indices.json](../data/splits/split_indices.json)
- [docs/xai/phase3b_protocol_v1.0.0.md](../docs/xai/phase3b_protocol_v1.0.0.md)
- [outputs/ablation/ablation_results.csv](../outputs/ablation/ablation_results.csv)
- [outputs/augmentation_experiment/experiment_config.json](../outputs/augmentation_experiment/experiment_config.json)
- [outputs/augmentation_experiment/rotation_results.csv](../outputs/augmentation_experiment/rotation_results.csv)
- [outputs/error_analysis/error_summary.json](../outputs/error_analysis/error_summary.json)
- [outputs/metrics/baseline_validation_evaluation.json](../outputs/metrics/baseline_validation_evaluation.json)
- [outputs/metrics/baseline_validation_metrics.json](../outputs/metrics/baseline_validation_metrics.json)
- [outputs/metrics/dataset_validation.json](../outputs/metrics/dataset_validation.json)
- [outputs/metrics/final_test_metrics.csv](../outputs/metrics/final_test_metrics.csv)
- [outputs/metrics/improved_training_history.csv](../outputs/metrics/improved_training_history.csv)
- [outputs/metrics/improved_validation_evaluation.json](../outputs/metrics/improved_validation_evaluation.json)
- [outputs/metrics/improved_validation_metrics.json](../outputs/metrics/improved_validation_metrics.json)
- [outputs/metrics/resnet_test_evaluation.json](../outputs/metrics/resnet_test_evaluation.json)
- [outputs/metrics/resnet_training_history.csv](../outputs/metrics/resnet_training_history.csv)
- [outputs/metrics/resnet_validation_evaluation.json](../outputs/metrics/resnet_validation_evaluation.json)
- [outputs/metrics/resnet_validation_metrics.json](../outputs/metrics/resnet_validation_metrics.json)
- [outputs/metrics/training_history.csv](../outputs/metrics/training_history.csv)
- [outputs/models/baseline_cnn_best.pth](../outputs/models/baseline_cnn_best.pth)
- [outputs/models/improved_cnn_best.pth](../outputs/models/improved_cnn_best.pth)
- [outputs/models/resnet_cnn_best.pth](../outputs/models/resnet_cnn_best.pth)
- [outputs/ood/external_compatibility_calibration.json](../outputs/ood/external_compatibility_calibration.json)
- [outputs/ood/ood_calibration.json](../outputs/ood/ood_calibration.json)
- [outputs/ood/ood_calibration.pt](../outputs/ood/ood_calibration.pt)
- [outputs/ood/ood_evaluation.json](../outputs/ood/ood_evaluation.json)
- [outputs/ood/ood_predictions.csv](../outputs/ood/ood_predictions.csv)
- [outputs/ood/pseudocolor_development_checks.json](../outputs/ood/pseudocolor_development_checks.json)
- [outputs/reproducibility/resnet_multiseed_results.csv](../outputs/reproducibility/resnet_multiseed_results.csv)
- [outputs/reproducibility/resnet_multiseed_summary.json](../outputs/reproducibility/resnet_multiseed_summary.json)
- [outputs/reproducibility_seed42_verify1/reproducibility/resnet_multiseed_summary.json](../outputs/reproducibility_seed42_verify1/reproducibility/resnet_multiseed_summary.json)
- [outputs/reproducibility_seed42_verify2/reproducibility/resnet_multiseed_summary.json](../outputs/reproducibility_seed42_verify2/reproducibility/resnet_multiseed_summary.json)
- [outputs/reproducibility_seed99_verify/reproducibility/resnet_multiseed_summary.json](../outputs/reproducibility_seed99_verify/reproducibility/resnet_multiseed_summary.json)
- [outputs/xai/gradcam_phase3a/metadata.json](../outputs/xai/gradcam_phase3a/metadata.json)
- [outputs/xai/phase3b_batch8/approval.json](../outputs/xai/phase3b_batch8/approval.json)
- [outputs/xai/phase3b_batch8/execution_status.json](../outputs/xai/phase3b_batch8/execution_status.json)
- [outputs/xai/phase3b_batch8/integrity_report.json](../outputs/xai/phase3b_batch8/integrity_report.json)
- [outputs/xai/phase3b_batch8/preflight.json](../outputs/xai/phase3b_batch8/preflight.json)
- [outputs/xai/phase3b_batch8/runner_stderr.jsonl](../outputs/xai/phase3b_batch8/runner_stderr.jsonl)
- [outputs/xai/phase3b_batch8/runner_stdout.json](../outputs/xai/phase3b_batch8/runner_stdout.json)
- [outputs/xai/phase3b_pilot_31_50/approval.json](../outputs/xai/phase3b_pilot_31_50/approval.json)
- [outputs/xai/phase3b_pilot_31_50/execution_status.json](../outputs/xai/phase3b_pilot_31_50/execution_status.json)
- [outputs/xai/phase3b_pilot_31_50/integrity_report.json](../outputs/xai/phase3b_pilot_31_50/integrity_report.json)
- [outputs/xai/phase3b_pilot_31_50/preflight.json](../outputs/xai/phase3b_pilot_31_50/preflight.json)
- [outputs/xai/phase3b_pilot_31_50/runner_stderr.jsonl](../outputs/xai/phase3b_pilot_31_50/runner_stderr.jsonl)
- [outputs/xai/phase3b_pilot_31_50/runner_stdout.json](../outputs/xai/phase3b_pilot_31_50/runner_stdout.json)
- [outputs/xai/phase3b_v1.0.0/real_validation/323bb0f19f5b8768/image_0031.json](../outputs/xai/phase3b_v1.0.0/real_validation/323bb0f19f5b8768/image_0031.json)
- [outputs/xai/phase3b_v1.0.0/real_validation/323bb0f19f5b8768/image_0050.json](../outputs/xai/phase3b_v1.0.0/real_validation/323bb0f19f5b8768/image_0050.json)
- [outputs/xai/phase3b_v1.0.0/real_validation/323bb0f19f5b8768/run.json](../outputs/xai/phase3b_v1.0.0/real_validation/323bb0f19f5b8768/run.json)
- [outputs/xai/phase3b_v1.0.0/real_validation/323bb0f19f5b8768/summary.json](../outputs/xai/phase3b_v1.0.0/real_validation/323bb0f19f5b8768/summary.json)
- [outputs/xai/phase3b_v1.0.0/real_validation/73452038829db01e/image_0035.json](../outputs/xai/phase3b_v1.0.0/real_validation/73452038829db01e/image_0035.json)
- [outputs/xai/phase3b_v1.0.0/real_validation/73452038829db01e/image_0037.json](../outputs/xai/phase3b_v1.0.0/real_validation/73452038829db01e/image_0037.json)
- [outputs/xai/phase3b_v1.0.0/real_validation/73452038829db01e/image_0038.json](../outputs/xai/phase3b_v1.0.0/real_validation/73452038829db01e/image_0038.json)
- [outputs/xai/phase3b_v1.0.0/real_validation/73452038829db01e/image_0051.json](../outputs/xai/phase3b_v1.0.0/real_validation/73452038829db01e/image_0051.json)
- [outputs/xai/phase3b_v1.0.0/real_validation/73452038829db01e/image_0052.json](../outputs/xai/phase3b_v1.0.0/real_validation/73452038829db01e/image_0052.json)
- [outputs/xai/phase3b_v1.0.0/real_validation/73452038829db01e/image_0053.json](../outputs/xai/phase3b_v1.0.0/real_validation/73452038829db01e/image_0053.json)
- [outputs/xai/phase3b_v1.0.0/real_validation/73452038829db01e/image_0055.json](../outputs/xai/phase3b_v1.0.0/real_validation/73452038829db01e/image_0055.json)
- [outputs/xai/phase3b_v1.0.0/real_validation/73452038829db01e/image_0094.json](../outputs/xai/phase3b_v1.0.0/real_validation/73452038829db01e/image_0094.json)
- [outputs/xai/phase3b_v1.0.0/real_validation/73452038829db01e/run.json](../outputs/xai/phase3b_v1.0.0/real_validation/73452038829db01e/run.json)
- [outputs/xai/phase3b_v1.0.0/real_validation/73452038829db01e/summary.json](../outputs/xai/phase3b_v1.0.0/real_validation/73452038829db01e/summary.json)
- [scripts/calibrate_ood_detector.py](../scripts/calibrate_ood_detector.py)
- [scripts/evaluate_test.py](../scripts/evaluate_test.py)
- [scripts/generate_gradcam.py](../scripts/generate_gradcam.py)
- [scripts/run_ablation_study.py](../scripts/run_ablation_study.py)
- [scripts/run_duplicate_audit_train_val.py](../scripts/run_duplicate_audit_train_val.py)
- [scripts/run_multiseed_resnet.py](../scripts/run_multiseed_resnet.py)
- [scripts/run_ood_evaluation.py](../scripts/run_ood_evaluation.py)
- [scripts/run_rotation_augmentation_experiment.py](../scripts/run_rotation_augmentation_experiment.py)
- [scripts/run_xai_validation_evaluation.py](../scripts/run_xai_validation_evaluation.py)
- [src/data/mirabest_pickle_stream_reader.py](../src/data/mirabest_pickle_stream_reader.py)
- [src/data/preprocess.py](../src/data/preprocess.py)
- [src/data/selective_raw_reader.py](../src/data/selective_raw_reader.py)
- [src/models/evaluate.py](../src/models/evaluate.py)
- [src/models/resnet_cnn.py](../src/models/resnet_cnn.py)
- [src/models/train.py](../src/models/train.py)
- [src/xai/evaluation_data.py](../src/xai/evaluation_data.py)
- [src/xai/execution_adapter.py](../src/xai/execution_adapter.py)
- [src/xai/faithfulness.py](../src/xai/faithfulness.py)
- [src/xai/gradcam.py](../src/xai/gradcam.py)
- [src/xai/morphology_audit.py](../src/xai/morphology_audit.py)
- [src/xai/sanity_checks.py](../src/xai/sanity_checks.py)
- [src/xai/stability.py](../src/xai/stability.py)
- [src/xai/statistics.py](../src/xai/statistics.py)
- [tests/test_duplicate_audit_train_val.py](../tests/test_duplicate_audit_train_val.py)
- [tests/test_mirabest_pickle_stream_reader.py](../tests/test_mirabest_pickle_stream_reader.py)
- [tests/test_selective_raw_reader.py](../tests/test_selective_raw_reader.py)
- [tests/test_xai_execution_adapter.py](../tests/test_xai_execution_adapter.py)
- [tests/test_xai_validation_runner.py](../tests/test_xai_validation_runner.py)
