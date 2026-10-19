"""Controlled Phase 3B runner: synthetic, metadata-only, or separately approved real.

Real execution requires an exact bounded approval receipt; disabled by default.
Four approved implementation gaps are not frozen-protocol fulfillment.
"""

import argparse
from copy import deepcopy
from dataclasses import asdict, fields, is_dataclass
from contextlib import contextmanager, nullcontext
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch import nn
import torchvision
import yaml

from src.data import preprocess as preprocessing
from src.data.preprocess import PreprocessConfig
from src.xai import evaluation_data as data
from src.xai import faithfulness as faith
from src.xai import gradcam, morphology_audit as morphology
from src.xai import sanity_checks as sanity, stability, statistics as stats

SCHEMA = "Phase3B-integration-v1.0.0"
UNAVAILABLE = {
    "secondary_stability_19x19": "native_grid_pooling_partial_support_ties_unspecified",
    "random_emission_baseline": "random_native_to_input_projection_unspecified",
    "sanity_brightness_map": "sanity_brightness_definition_unspecified",
    "sanity_random_map": "sanity_random_distribution_grid_seeds_replicates_projection_unspecified",
}
NOTICE = "Synthetic integration only; no scientific findings or real validation-image evaluation."
SYNTHETIC_PROFILE = {"mode": "synthetic", "namespace": "synthetic_integration",
                     "origin": "generated_synthetic_fixture", "notice": NOTICE,
                     "scientific_validation_evaluated": False}

RESOURCE_FAMILIES = ("original_prediction_gradcam", "primary_aopc", "full_deletion_curves_auc",
                     "stability_robustness", "radio_emission_alignment", "parameter_randomization",
                     "statistical_aggregation", "frozen_ood_diagnostics")


class ResourceSafetyError(RuntimeError):
    """A cooperative stop; never an asynchronous/hard interruption guarantee."""


class PilotResources:
    """Operational telemetry only; no values enter scientific records or run IDs.

    2 GiB/900 s are conservative pilot stop thresholds, not measured hardware
    requirements. Checks cannot interrupt a blocking forward, backward, OS call,
    or a mathematical API that exposes no callback. Peak memory is process-wide
    lifetime peak; working-set enforcement applies at sampled checkpoints.
    """

    def __init__(self, *, max_working_set_bytes=2*1024**3, max_image_seconds=900,
                 memory_reader=None, clock=None):
        if (type(max_working_set_bytes) is not int or max_working_set_bytes <= 0 or
                isinstance(max_image_seconds, bool) or not isinstance(max_image_seconds, (int, float)) or
                not math.isfinite(max_image_seconds) or max_image_seconds <= 0):
            raise ValueError("Pilot resource limits must be positive and finite.")
        if memory_reader is None:
            from src.xai.execution_adapter import memory_snapshot
            memory_reader = memory_snapshot
        self.memory_reader, self.clock = memory_reader, clock or time.perf_counter
        self.max_bytes, self.max_seconds = max_working_set_bytes, max_image_seconds
        self.started = self.clock()
        self.families = {name: self._record() for name in RESOURCE_FAMILIES}
        self.samples, self.max_sampled_bytes, self.process_peak_bytes = 0, 0, None
        self.stack, self.failure, self.stop_code = [], None, None
        self.last_memory = {"status": "not_sampled"}
        self.check()

    @staticmethod
    def _record():
        return {"elapsed_seconds": 0.0, "inclusive_seconds": 0.0, "calls": 0, "failures": 0}

    def _violate(self, code, message):
        self.stop_code = self.stop_code or code
        self.failure = self.failure or {"reason_code": code,
                                        "metric_family": self.stack[-1]["name"] if self.stack else None}
        error = ResourceSafetyError(message)
        error.reason_code = self.stop_code
        error.pilot_resource_report = self.report()
        raise error

    def check(self):
        if self.stop_code is not None:
            self._violate(self.stop_code, "Execution remains stopped after a resource violation.")
        elapsed = self.clock()-self.started
        if not math.isfinite(elapsed) or elapsed < 0:
            self._violate("unreliable_clock", "Unreliable elapsed-time measurement.")
        try:
            snapshot = self.memory_reader()
        except Exception:
            snapshot = None  # Never copy an OS exception/path into telemetry.
        working = snapshot.get("working_set_bytes") if isinstance(snapshot, dict) else None
        if type(working) is not int or working <= 0:
            self.last_memory = {"status": "unavailable", "reason_code": "memory_measurement_unavailable"}
            self._violate("memory_measurement_unavailable", "Reliable process working-set measurement unavailable; fail closed.")
        self.last_memory = {"status": "available", "working_set_bytes": working, "sample_elapsed_seconds": elapsed}
        self.samples += 1
        self.max_sampled_bytes = max(working, self.max_sampled_bytes)
        peak = snapshot.get("process_peak_bytes")
        if type(peak) is int and peak >= working:
            self.process_peak_bytes = max(peak, self.process_peak_bytes or 0)
            self.last_memory["process_lifetime_peak_bytes"] = peak
        if elapsed > self.max_seconds:
            self._violate("time_limit_exceeded", "Pilot per-image elapsed-time limit exceeded at a cooperative checkpoint.")
        if working > self.max_bytes:
            self._violate("memory_limit_exceeded", "Pilot process working-set limit exceeded.")

    def _begin(self, name):
        frame = {"name": name, "start": self.clock(), "children": 0.0}
        self.families.setdefault(name, self._record())["calls"] += 1
        self.stack.append(frame)
        return frame

    def _end(self, frame, *, failed=False):
        if not self.stack or self.stack[-1] is not frame:
            raise RuntimeError("Operational timer stack mismatch.")
        duration = self.clock()-frame["start"]
        self.stack.pop()
        record = self.families[frame["name"]]
        # Exclusive durations partition instrumented work. Inclusive duration is
        # a UNION for each family: recursive/same-family calls are counted once.
        record["elapsed_seconds"] += max(0.0, duration-frame["children"])
        if not any(parent["name"] == frame["name"] for parent in self.stack):
            record["inclusive_seconds"] += duration
        record["failures"] += int(failed)
        if self.stack:
            self.stack[-1]["children"] += duration

    @contextmanager
    def family(self, name):
        try:
            self.check()
        except Exception:
            if self.failure is not None and self.failure["metric_family"] is None:
                self.failure["metric_family"] = name
            raise
        frame, failed = self._begin(name), False
        try:
            yield
            self.check()
        except BaseException:
            failed = True
            self.failure = self.failure or {"reason_code": "execution_error", "metric_family": name}
            raise
        finally:
            self._end(frame, failed=failed)

    def call(self, name, function, *args, **kwargs):
        with self.family(name):
            return function(*args, **kwargs)

    def scorer(self, function):
        def guarded(tensor):
            self.check()
            result = function(tensor)
            self.check()
            return result
        return guarded

    def report(self):
        elapsed = self.clock()-self.started
        return {"elapsed_seconds": elapsed if math.isfinite(elapsed) and elapsed >= 0 else None,
                "families": deepcopy(self.families), "failure": deepcopy(self.failure),
                "active_metric_family": self.stack[-1]["name"] if self.stack else None,
                "last_memory_measurement": deepcopy(self.last_memory),
                "instrumented_exclusive_seconds": sum(r["elapsed_seconds"] for r in self.families.values()),
                "limits": {"max_working_set_bytes": self.max_bytes, "max_image_seconds": self.max_seconds},
                "memory_samples": self.samples, "max_sampled_working_set_bytes": self.max_sampled_bytes,
                "process_lifetime_peak_bytes": self.process_peak_bytes,
                "peak_status": "available" if self.process_peak_bytes is not None else "unavailable",
                "enforcement": "cooperative checkpoints; blocking operations cannot be interrupted immediately",
                "timing_scope": "elapsed_seconds per family is exclusive and additive across families. inclusive_seconds is a per-family union, not additive across families. Uninstrumented overhead is excluded. Primary AOPC reuses deletion forwards; its timer measures derived calculation only. OOD timer observes validate_tensor, excluding the paired classifier call.",
                "sanity_checkpoints": "before/after family only; mathematical API exposes no stage callback"}

    def ood_scorer(self, function):
        """Observe the existing diagnostic call without replacing mathematical APIs.

        A temporary thread-local Python profile callback times only validate_tensor.
        Restore the caller's profiler on every exit; refuse an existing profiler.
        """
        from src.ood.ood_detector import OODDetector
        code = OODDetector.validate_tensor.__code__

        def score(tensor):
            self.check()
            previous = sys.getprofile()
            if previous is not None:
                raise ResourceSafetyError("Existing Python profiler conflicts with isolated OOD timing.")
            entered = {}

            def observe(frame, event, arg):
                if frame.f_code is code:
                    if event == "call":
                        self.check()
                        entered[id(frame)] = self._begin("frozen_ood_diagnostics")
                    elif event == "return":
                        timer = entered.pop(id(frame), None)
                        if timer is not None:
                            self._end(timer, failed=arg is None)
                        self.check()

            try:
                sys.setprofile(observe)
                return function(tensor)
            finally:
                sys.setprofile(previous)
                self.check()
        return score


def _timed(resources, name, function, *args, **kwargs):
    return resources.call(name, function, *args, **kwargs) if resources is not None else function(*args, **kwargs)


def _family(resources, name):
    return resources.family(name) if resources is not None else nullcontext()


def _failure_diagnostics(error, resources=None):
    """Allowlisted operational fields only: never exception text/traceback/paths."""
    resource_codes = {"memory_limit_exceeded", "time_limit_exceeded", "memory_measurement_unavailable",
                      "unreliable_clock"}
    code = getattr(error, "reason_code", None)
    if code not in resource_codes:
        code = ("resource_safety_failure" if isinstance(error, ResourceSafetyError) else
                "unauthorized_execution" if isinstance(error, PermissionError) else
                "publication_conflict" if isinstance(error, FileExistsError) else
                "invalid_execution_data_or_provenance" if isinstance(error, ValueError) else "execution_error")
    report = resources.report() if resources is not None else getattr(error, "pilot_resource_report", None)
    if report is None:
        report = {"elapsed_seconds": None, "active_metric_family": None,
                  "last_memory_measurement": {"status": "not_sampled", "reason_code": "failure_before_resource_monitor"}}
    return {"reason_code": code, "message": "Execution stopped; scientific completion was not reported.",
            "resource_report": report}


LIMITATIONS = (
    "Validation is exploratory and has informed model development.",
    "Candidate-source identity and source-level independence are not independently verified.",
    "Sign-flip inference assumes appropriate paired-null symmetry/exchangeability.",
    "Intervals condition on metric availability, not retrained-model or external-survey uncertainty.",
    "Emission masks are processed-intensity proxies, not verified jets/lobes/cores/hotspots.",
    "Preserved learned BatchNorm buffers mean final randomized models are not wholly fresh networks.",
    "Approved unavailable analyses remain incomplete; no metric proves causality or astrophysical correctness.",
)
PROTECTED_ROOTS = ("outputs/models", "outputs/metrics", "outputs/ood", "data/splits")


def primitive(value):
    """Canonical finite JSON representation; never serialize full image tensors."""
    if is_dataclass(value):
        return {f.name: primitive(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {str(k): primitive(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [primitive(v) for v in value]
    if isinstance(value, np.generic):
        return primitive(value.item())
    if isinstance(value, (torch.Tensor, np.ndarray)):
        raise ValueError("Image arrays/tensors must not enter result serialization.")
    return value


def canonical(value):
    return data.canonical_bytes(primitive(value))


def protected_hashes(root):
    return {str(p.relative_to(root)).replace("\\", "/"): data.digest(p.read_bytes())
            for folder in PROTECTED_ROOTS for p in sorted((root / folder).rglob("*")) if p.is_file()}


def audit_api_contract(config):
    """Fail closed on executable/configuration drift; no protocol amendment."""
    f, s, t, r, st = (config[k] for k in ("faithfulness", "stability", "transformations", "randomization", "statistics"))
    checks = (
        config["protocol"] == data.PROTOCOL == faith.PROTOCOL == stability.PROTOCOL,
        tuple(f["fractions"]) == faith.FULL_FRACTIONS,
        tuple(f["fractions"][1:7]) == faith.FRACTIONS and f["primary_steps"] == 6,
        tuple(f["random_seeds"]) == faith.RANDOM_SEEDS,
        tuple(t["rotations_degrees"]) == stability.ROTATIONS,
        tuple(t["sensitivity_rotations_degrees"]) == stability.SENSITIVITY_ROTATIONS,
        tuple(t["noise_seeds"]) == stability.NOISE_SEEDS,
        tuple(t["gaussian_noise_std"]) == stability.NOISE_SIGMAS,
        tuple(t["brightness_gains"]) == stability.BRIGHTNESS_GAINS,
        (s["primary_top_fraction"], *s["sensitivity_top_fractions"]) == stability.TOP_FRACTIONS,
        s["primary_grid"] == [150, 150] and s["minimum_valid_fraction"] == .8,
        tuple(r["seeds"]) == sanity.SEEDS and tuple(r["cumulative_order"]) == sanity.STAGES,
        st["bootstrap"]["replicates"] == st["permutation"]["replicates"] == stats.REPLICATES,
        st["bootstrap"]["seed"] == stats.BOOTSTRAP_SEED and st["permutation"]["seed"] == stats.PERMUTATION_SEED,
        tuple(st["inferential_family"]) == stats.PRIMARY_FAMILY,
        st["correction"]["familywise_alpha"] == stats.ALPHA,
        config["gradcam"]["target_layer"] == gradcam.TARGET_LAYER,
        config["sampling"]["class_mapping"] == {0: "FRI", 1: "FRII"},
        PreprocessConfig().normalize_mean == config["preprocessing"]["mean"] == .5,
        PreprocessConfig().normalize_std == config["preprocessing"]["std"] == .5,
        morphology.random_control_availability().reason == UNAVAILABLE["random_emission_baseline"],
    )
    if not all(checks):
        raise ValueError("Frozen protocol/API conflict.")


class VerifiedContext:
    """Own the checked metadata boundary; never expose an image dataset."""

    def __init__(self, root=PROJECT_ROOT):
        self.root = Path(root).resolve()
        self.access = data.load_evaluation_data(self.root)
        config = yaml.safe_load((self.root / data.CONFIG_PATH).read_bytes())
        if data.digest(canonical(config)) != data.CONFIG_SHA256:
            raise ValueError("Frozen protocol configuration mismatch.")
        audit_api_contract(config)
        entries = self.access.validation.entries
        if (len(entries) != 119 or len(self.access.validation_allowlist) != 119 or
                tuple(e.dataset_index for e in entries) != tuple(sorted(self.access.validation_allowlist)) or
                any(type(e.ground_truth_label) is not int or e.ground_truth_label not in (0, 1) or
                    e.split_membership != "validation" for e in entries)):
            raise ValueError("Exact frozen validation allowlist/labels required.")
        self.protected = protected_hashes(self.root)
        modules = (data, faith, gradcam, stability, morphology, sanity, stats, preprocessing)
        self.metadata = {
            "schema": SCHEMA, "protocol": data.PROTOCOL, "notice": NOTICE,
            "provenance": asdict(self.access.validation.provenance),
            "validation_manifest_sha256": self.access.validation.sha256,
            "engineering_manifest_sha256": self.access.engineering.sha256,
            "protected_artifacts": self.protected,
            "implementation_sha256": {m.__name__: data.digest(Path(m.__file__).read_bytes()) for m in modules},
            "runner_sha256": data.digest(Path(__file__).read_bytes()),
            "software": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                         "torchvision": torchvision.__version__, "pyyaml": yaml.__version__},
            "synthetic_execution": {"device": "cpu", "torch_threads": 1, "model_seed": 42},
            "approved_unavailable": UNAVAILABLE,
            "complete_frozen_protocol_coverage": False,
            "implemented_metrics": ["primary_AOPC30", "full_deletion_curves_AUC", "aligned_150x150_stability",
                                    "intensity_emission_proxy_alignment", "six_stage_parameter_randomization",
                                    "identity_repeated_frozen_CAM", "image_level_statistics"],
            "real_image_evaluation_enabled": False,
            "scientific_limitations": LIMITATIONS,
        }

    def authorize(self, index):
        entry = self.access.authorize(index, "validation")
        if entry not in self.access.validation.entries or entry.ground_truth_label not in (0, 1):
            raise ValueError("Validation manifest drift.")
        self.check_protected()
        return entry

    def check_protected(self):
        if protected_hashes(self.root) != self.protected:
            raise ValueError("Protected artifact drift; abort and invalidate run.")


def workload():
    return {
        "eligible_validation_images": 119, "primary_deletion_forwards_uncached": 119*22*6,
        "full_three_replacement_deletion_forwards": 119*22*20*3,
        "primary_reuses_full_first_six_points": True,
        "full_api_original_forwards": 119*3,
        "original_and_transformed_cam_calls": 119*31,
        "randomized_stage_cam_calls": 119*30,
        "sanity_reference_and_identity_cam_calls": 119*2,
        "total_cam_calls_with_current_controls": 119*63,
        "raw_curve_logit_float64_lower_bound_bytes": 119*3*22*(21+6)*2*8,
        "runtime": "Unmeasured. CAM requires backward passes; sorting, hashing and OOD diagnostics add work. No real-image timing probe performed.",
        "storage": "Stream images/models. Protocol planning RAM: 2 GB starting budget, unmeasured. Retaining all CAMs ~1.3 GB and deletion images >14 GB; neither is saved. Scalar/logit JSON size is reported only for synthetic records, not extrapolated as measured research storage.",
    }


def dry_run(root=PROJECT_ROOT):
    context = VerifiedContext(root)
    entries = context.access.validation.entries
    return {
        **context.metadata, "mode": "dry-run", "validation_indices": [e.dataset_index for e in entries],
        "validation_class_counts": {str(c): sum(e.ground_truth_label == c for e in entries) for c in (0, 1)},
        "engineering_count": len(context.access.engineering.entries),
        "implementation_coverage": {k: {"status": "unavailable", "reason": v, "eligible_images": 119}
                                    for k, v in UNAVAILABLE.items()},
        "scientific_validation_coverage": {"eligible": 119, "evaluated": 0},
        "image_pixels_decoded": 0, "model_forwards": 0,
        "locked_test_images_accessed": 0, "workload_estimate": workload(),
    }


class _SyntheticBlock(nn.Module):
    def __init__(self, incoming, outgoing):
        super().__init__()
        self.convolution = nn.Sequential(nn.Conv2d(incoming, 2, 1), nn.BatchNorm2d(2),
                                          nn.ReLU(), nn.Conv2d(2, outgoing, 1), nn.BatchNorm2d(outgoing))
        self.shortcut = nn.Sequential(nn.Conv2d(incoming, outgoing, 1), nn.BatchNorm2d(outgoing))

    def forward(self, x):
        return (self.convolution(x)+self.shortcut(x)).relu()


class _SyntheticModel(nn.Module):
    """Disposable lightweight mock, NOT the frozen research ResNet."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.AdaptiveAvgPool2d((19, 19)), nn.Conv2d(1, 2, 1), nn.BatchNorm2d(2), nn.ReLU())
        self.blocks = nn.Sequential(_SyntheticBlock(2, 2), _SyntheticBlock(2, 2), _SyntheticBlock(2, 128))
        self.spatial_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(512, 4), nn.ReLU(), nn.Dropout(.3), nn.Linear(4, 2))
        # Analytically positive paths give a nonconstant synthetic reference CAM.
        # This fixture convention has no application to research model weights.
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    module.weight.abs_()
                    if module.bias is not None:
                        module.bias.zero_()
            self.classifier[4].weight[1].neg_()

    def forward(self, x):
        return self.classifier(self.spatial_pool(self.blocks(self.stem(x))))


def _synthetic_image(index):
    y, x = np.mgrid[:150, :150]
    return (.1+.2*x/149+.1*y/149+.4*np.exp(-((x-45-index%7)**2+(y-85)**2)/200)).astype(np.float32)


def _state(model):
    return (tuple((n, t.detach().clone()) for n, t in model.state_dict().items()),
            tuple((p.requires_grad, None if p.grad is None else p.grad.detach().clone()) for p in model.parameters()),
            tuple(m.training for m in model.modules()))


def _same_state(first, second):
    return (len(first[0]) == len(second[0]) and len(first[1]) == len(second[1]) and
            first[2] == second[2] and all(a[0] == b[0] and torch.equal(a[1], b[1]) for a, b in zip(first[0], second[0]))
            and all(a[0] == b[0] and ((a[1] is None and b[1] is None) or
                    (a[1] is not None and b[1] is not None and torch.equal(a[1], b[1]))) for a, b in zip(first[1], second[1])))


def _curve_payload(full, resources=None):
    curves = (("gradcam", None, full.gradcam), ("brightness", None, full.brightness),
              *(("random", seed, curve) for seed, curve in full.random))
    primary, secondary = [], []
    scores = {}
    for name, seed, curve in curves:
        six = faith.DeletionCurve(curve.target_class, curve.original_logits, curve.logits[1:7])
        value = _timed(resources, "primary_aopc", faith.aopc, six)
        scores[(name, seed)] = value
        primary.append({"ranking": name, "seed": seed, "curve": six, "aopc": value})
        secondary.append({"ranking": name, "seed": seed, "curve": curve,
                          "auc": _timed(resources, "full_deletion_curves_auc", faith.deletion_auc, curve)})
    contrast = _timed(resources, "primary_aopc", faith.paired_contrasts, scores[("gradcam", None)], scores[("brightness", None)],
                                     {s: scores[("random", s)] for s in faith.RANDOM_SEEDS})
    return {"status": "available", "primary": primary, "secondary": secondary, "primary_contrasts": contrast}, scores


def _independent_controls(image, rankings, scorer, replacement, original, target, resources=None):
    """No fabricated CAM rank when reference attribution is degenerate."""
    replaced = faith.replacement_image(image, replacement)
    output = []
    for name, seed, order in (("brightness", None, rankings[0]),
                              *(("random", seed, order) for seed, order in rankings[1].items())):
        logits = [original]
        for mask in faith.full_deletion_masks(order)[1:]:
            tensor = faith.normalize_intensity(faith.perturb_image(image, mask, replaced))
            z = tuple(scorer(tensor)[0].tolist())
            faith.target_scores(z, target)  # validate existing finite-score contract
            logits.append(z)
        curve = faith.FullDeletionCurve(target, original, tuple(logits))
        six = faith.DeletionCurve(target, original, tuple(logits[1:7]))
        output.append({"ranking": name, "seed": seed, "curve": curve,
                       "auc": _timed(resources, "full_deletion_curves_auc", faith.deletion_auc, curve),
                       "aopc": _timed(resources, "primary_aopc", faith.aopc, six)})
    return output


def evaluate_synthetic(context, index, model):
    """Authorize BEFORE generating arrays/forwarding. No caller image input API."""
    entry = context.authorize(index)
    if type(model) is not _SyntheticModel:
        raise PermissionError("Only the generated synthetic mock model is permitted.")
    return _evaluate_image(context, index, model, _synthetic_image(index), entry)


def _evaluate_image(context, index, model, image, entry, *, profile=None, scorer=None, diagnostics=None, resources=None):
    """Shared metric orchestration; callers own authorization and data isolation."""
    if entry != context.authorize(index):
        raise ValueError("Authorized entry mismatch.")
    profile = profile or SYNTHETIC_PROFILE
    before = _state(model)
    original_image = image.copy()
    metrics = {name: stats.observation(reason=reason) for name, reason in UNAVAILABLE.items()}
    result = {"schema": SCHEMA, "origin": profile["origin"], "notice": profile["notice"],
              "dataset_index": index, "ground_truth_label": entry.ground_truth_label,
              "scientific_validation_evaluated": profile["scientific_validation_evaluated"], "provenance": asdict(entry)}

    def measure(name, value, reasons=()):
        metrics[name] = stats.observation(value) if value is not None else stats.observation(reason=";".join(reasons) or "unavailable")

    try:
        original = _timed(resources, "original_prediction_gradcam", gradcam.generate_gradcam,
                          model, faith.normalize_intensity(image))
        target = original.predicted_class
        if original.target_class != target:
            raise ValueError("Original CAM target drift.")
        z = tuple(original.logits[0].tolist())
        original_scores = faith.target_scores(z, target)
        for field in ("probability", "logit", "margin"):
            measure("original_class."+field, getattr(original_scores, field))
        result.update(predicted_class=target, fixed_target_class=target, logits=z,
                      confidence=original_scores.probability, classification_correct=target == entry.ground_truth_label,
                      explanation_status=original.status, layer=original.layer_name,
                      native_shape=list(original.raw_cam.shape), heatmap_shape=list(original.upsampled_heatmap.shape))
        scorer = scorer or faith.ModelScorer(model)
        if resources is not None:
            scorer = resources.scorer(scorer)
        result["faithfulness"] = {}
        rankings = faith.prepare_rankings(image, original.raw_cam.numpy(), index) if original.status == "valid" else None
        controls = (faith.rank_brightness(image, index), faith.random_rankings(index)) if rankings is None else None
        for policy in faith.REPLACEMENTS:
            if rankings is not None:
                full = _timed(resources, "full_deletion_curves_auc", faith.evaluate_full_rankings,
                              image, rankings, scorer, replacement=policy)
                if full.gradcam.target_class != target or full.gradcam.original_logits != z:
                    raise ValueError("Deletion target/logit drift.")
                payload, values = _curve_payload(full, resources)
            else:
                payload = {"status": "unavailable", "reasons": ["original_gradcam_degenerate"],
                           "independent_controls": _timed(resources, "full_deletion_curves_auc", _independent_controls,
                                                          image, controls, scorer, policy, z, target, resources)}
                values = {(row["ranking"], row["seed"]): row["aopc"] for row in payload["independent_controls"]}
            result["faithfulness"][policy] = payload
            for comparison in ("cam_minus_random", "cam_minus_brightness"):
                for field in ("probability", "logit", "margin"):
                    value = (getattr(getattr(payload["primary_contrasts"], comparison), field)
                             if rankings is not None else None)
                    measure(f"faithfulness.{policy}.{field}.{comparison}", value, ("original_gradcam_degenerate",))
            for field in ("probability", "logit", "margin"):
                prefix = f"faithfulness.{policy}.{field}"
                for name in ("gradcam", "brightness"):
                    value = values.get((name, None))
                    measure(prefix+"."+name, getattr(value, field) if value else None, ("original_gradcam_degenerate",))
                metrics[prefix+".random"] = _timed(resources, "statistical_aggregation", stats.aggregate_repeats,
                    {s: getattr(values[("random", s)], field) for s in faith.RANDOM_SEEDS}, "random_deletion")
            for row in payload.get("secondary", payload.get("independent_controls", [])):
                for field in ("probability", "logit", "margin"):
                    key = f"auc.{policy}.{field}.{row['ranking']}"
                    if row["ranking"] != "random":
                        measure(key, getattr(row["auc"], field))
            auc_rows = payload.get("secondary", payload.get("independent_controls", []))
            for field in ("probability", "logit", "margin"):
                if rankings is None:
                    measure(f"auc.{policy}.{field}.gradcam", None, ("original_gradcam_degenerate",))
                metrics[f"auc.{policy}.{field}.random"] = _timed(resources, "statistical_aggregation", stats.aggregate_repeats,
                    {row["seed"]: getattr(row["auc"], field) for row in auc_rows if row["ranking"] == "random"},
                    "random_deletion")
            if policy == "border_median":
                metrics[stats.CAM] = metrics[f"faithfulness.{policy}.probability.gradcam"]
                metrics[stats.BRIGHTNESS] = metrics[f"faithfulness.{policy}.probability.brightness"]
                metrics[stats.RANDOM] = metrics[f"faithfulness.{policy}.probability.random"]

        with _family(resources, "stability_robustness"):
            comparisons = []
            for spec in stability.protocol_transformations():
                if resources is not None:
                    resources.check()
                transformed = stability.apply_transform(image, spec, index)
                explanation = gradcam.generate_gradcam(model, faith.normalize_intensity(transformed.image), target)
                comparisons.append(stability.compare_explanations(original, explanation, transformed, index))
            result["stability"] = comparisons
            result["stability_case_counts"] = {k: len(v) for k, v in stability.partition_cases(comparisons).items()}
            for comparison in comparisons:
                spec = comparison.transformation
                prefix = f"stability.{spec.kind}.{spec.value}"
                if spec.kind != "noise":
                    for j in comparison.jaccard:
                        measure(f"{prefix}.top{j.fraction}", j.similarity, (comparison.reason,))
                        for group in ("prediction_preserving", "class_flipped"):
                            measure(f"{prefix}.{group}.top{j.fraction}",
                                    j.similarity if comparison.case_group == group else None,
                                    (comparison.reason,) if comparison.case_group == group else ("other_prediction_case",))
                    measure(prefix+".probability_change", comparison.original_class_probability_change)
                    measure(prefix+".margin_change", comparison.original_class_margin_change)
                    for group in ("prediction_preserving", "class_flipped"):
                        for name, value in (("probability_change", comparison.original_class_probability_change),
                                            ("margin_change", comparison.original_class_margin_change)):
                            measure(f"{prefix}.{group}.{name}", value if comparison.case_group == group else None,
                                    ("other_prediction_case",))
            for sigma in stability.NOISE_SIGMAS:
                rows = [r for r in comparisons if r.transformation.kind == "noise" and r.transformation.value == sigma]
                for fraction in stability.TOP_FRACTIONS:
                    values = {r.transformation.seed: stats.observation(next(j.similarity for j in r.jaccard if j.fraction == fraction),
                             reason=r.reason if r.status != "valid" else None) for r in rows}
                    metrics[f"stability.noise.{sigma}.top{fraction}"] = _timed(resources, "statistical_aggregation", stats.aggregate_repeats, values, "noise", sigma)
                    for group in ("prediction_preserving", "class_flipped"):
                        metrics[f"stability.noise.{sigma}.{group}.top{fraction}"] = _timed(resources, "statistical_aggregation", stats.aggregate_repeats,
                            {r.transformation.seed: values[r.transformation.seed] if r.case_group == group else
                             stats.observation(reason="other_prediction_case") for r in rows}, "noise", sigma)
                for field in ("original_class_probability_change", "original_class_margin_change"):
                    metrics[f"stability.noise.{sigma}.{field}"] = _timed(resources, "statistical_aggregation", stats.aggregate_repeats,
                        {r.transformation.seed: getattr(r, field) for r in rows}, "noise", sigma)
                    for group in ("prediction_preserving", "class_flipped"):
                        metrics[f"stability.noise.{sigma}.{group}.{field}"] = _timed(resources, "statistical_aggregation", stats.aggregate_repeats,
                            {r.transformation.seed: getattr(r, field) if r.case_group == group else
                             stats.observation(reason="other_prediction_case") for r in rows}, "noise", sigma)

        with _family(resources, "radio_emission_alignment"):
            alignment = morphology.audit_emission_alignment(image, original.upsampled_heatmap, index, cam_status=original.status)
            result["emission"] = []
            for variant in alignment.variants:
                measure(f"emission.{variant.proxy.variant}.area", variant.proxy.emission_area_fraction)
                measure(f"emission.{variant.proxy.variant}.threshold", variant.proxy.absolute_threshold)
                proxy = {f.name: getattr(variant.proxy, f.name) for f in fields(variant.proxy) if f.name != "emission_mask"}
                result["emission"].append({"proxy": proxy, "gradcam": variant.gradcam,
                                           "brightness": variant.brightness, "uniform": variant.uniform, "random": variant.random})
                for kind in ("gradcam", "brightness", "uniform"):
                    overlap = getattr(variant, kind)
                    for field in ("emission_attribution_mass", "background_attribution_mass", "area_adjusted_enrichment"):
                        measure(f"emission.{variant.proxy.variant}.{kind}.{field}", getattr(overlap, field), overlap.reasons)
                    measure(f"emission.{variant.proxy.variant}.{kind}.top10_iou", overlap.top10_emission_iou, overlap.top10_reasons)

        randomized = _timed(resources, "parameter_randomization", sanity.run_sanity_checks,
                            model, faith.normalize_intensity(image), index)
        if any(r.fixed_target_class != target for r in randomized):
            raise ValueError("Randomization target drift.")
        result["sanity"] = randomized
        for field in ("jaccard", "spearman", "total_variation"):
            identity = randomized[0].similarity
            measure("sanity.identity."+field, getattr(identity, field), identity.reasons)
        for stage in range(1, 7):
            for field in ("jaccard", "spearman", "total_variation"):
                metrics[f"sanity.stage{stage}.{field}"] = _timed(resources, "statistical_aggregation", stats.aggregate_repeats,
                    {r.seed: stats.observation(getattr(r.similarity, field),
                         reason=";".join(r.similarity.reasons) if r.similarity.status != "valid" else None)
                     for r in randomized if r.stage == stage}, "randomization", stage)
        result["record"] = _timed(resources, "statistical_aggregation", stats.image_record,
                                   index, entry.ground_truth_label, target, metrics)
        result["approved_unavailable"] = UNAVAILABLE
        result["ood_diagnostics"] = {"status": "not_executed_synthetic",
                                     "reason": "frozen_research_calibration_does_not_apply_to_mock_model",
                                     "future_requirement": "Record frozen diagnostics without filtering deletion curves."}
        if diagnostics is not None:
            result["ood_diagnostics"] = {"status": "available", "scope": "deletion scoring calls in recorded curve order",
                                         "classifier_curves_filtered": False, "calls": diagnostics}
        return primitive(result)
    finally:
        if not np.array_equal(image, original_image) or not _same_state(before, _state(model)):
            raise RuntimeError("Input/model state drift; invalidate run.")
        context.check_protected()
        if resources is not None:
            resources.check()


def _restore_observation(raw):
    return stats.Observation(raw["value"], raw["status"], tuple(raw["reasons"]), raw["repeat_kind"], raw["setting"],
                             tuple((seed, _restore_observation(value)) for seed, value in raw["repeats"]))


def restore_record(payload):
    r = payload["record"]
    return stats.image_record(r["dataset_index"], r["ground_truth_label"], r["predicted_class"],
                              {name: _restore_observation(value) for name, value in r["metrics"]})


def summarize_results(payloads, *, profile=None):
    profile = profile or SYNTHETIC_PROFILE
    records = tuple(restore_record(p) for p in payloads)
    # Project only the relevant metrics into each existing statistical API.
    # This avoids repeatedly validating hundreds of unrelated seed groups.
    # Complete original per-image records, raw repetitions and pairs stay stored.
    metric_maps = {r.dataset_index: dict(r.metrics) for r in records}
    primary_inputs = tuple(stats.image_record(r.dataset_index, r.ground_truth_label, r.predicted_class,
                           {k: v for k, v in metric_maps[r.dataset_index].items()
                            if k in (stats.CAM, stats.BRIGHTNESS, stats.RANDOM)}) for r in records)
    analysis = stats.analyze_primary(primary_inputs)
    for record in analysis.records:
        metric_maps[record.dataset_index].update({k: v for k, v in record.metrics if k in stats.PRIMARY_FAMILY})
    names = sorted(set(UNAVAILABLE) | {name for values in metric_maps.values() for name in values})
    summaries = {}
    for name in names:
        views = tuple(stats.image_record(r.dataset_index, r.ground_truth_label, r.predicted_class,
                                        {name: metric_maps[r.dataset_index][name]}
                                        if name in metric_maps[r.dataset_index] else {}) for r in records)
        summaries[name] = stats.summarize(views, name)
    output = {"schema": SCHEMA, "notice": NOTICE, "mode": "synthetic", "synthetic_images_completed": len(records),
              "complete_frozen_protocol_coverage": False,
              "scientific_limitations": LIMITATIONS,
              "scientific_validation_coverage": {"eligible": 119, "evaluated": 0},
              "metric_summaries": summaries,
              "stability_case_counts": {group: sum(p["stability_case_counts"][group] for p in payloads)
                                        for group in ("prediction_preserving", "class_flipped")},
              "primary_tests": analysis.tests, "holm": analysis.holm,
              "bootstrap": [{f.name: getattr(b, f.name) for f in fields(b)
                              if f.name not in {"macro_replicates", "overall_replicates"}} for b in analysis.bootstrap],
              "approved_unavailable": UNAVAILABLE, "workload_estimate": workload()}
    sizes = [len(canonical(p)) for p in payloads]
    if profile["mode"] == "real":
        output.update(notice=profile["notice"], mode="real", validation_images_completed=len(records),
                      scientific_validation_coverage={"eligible": 119, "evaluated": len(records)})
        del output["synthetic_images_completed"]
    output["synthetic_storage_planning"] = {
        "record_bytes": sizes, "119_record_layout_estimate_bytes": max(sizes)*119 if sizes else None,
        "basis": "Synthetic JSON layout only; not a real-image storage measurement. Summary/manifest overhead excluded."}
    return primitive(output)


class ResultStore:
    """No-overwrite atomic hard-link publication; fail closed on partial files.

    A crash leaving .pending files requires inspection; it is never guessed to be
    completed work. Hard-link support is required; no unsafe overwrite fallback.
    Digests detect corruption, not malicious re-signing or provide authentication.
    """

    def __init__(self, context, indices, *, resume=False, profile=None):
        self.context = context
        self.profile = profile or SYNTHETIC_PROFILE
        if self.profile["mode"] not in {"real", "synthetic"}:
            raise ValueError("Unsupported execution profile.")
        expected_namespace = "real_validation" if self.profile["mode"] == "real" else "synthetic_integration"
        if self.profile["namespace"] != expected_namespace:
            raise PermissionError("Execution namespace mismatch.")
        self.indices = tuple(indices)
        if len(set(self.indices)) != len(self.indices) or tuple(sorted(self.indices)) != self.indices:
            raise ValueError("Duplicate or unsorted requested image IDs.")
        for index in self.indices:
            context.authorize(index)
        if self.profile["mode"] == "real":
            from src.xai.execution_adapter import approval_request
            request = self.profile["execution"]
            if request != approval_request(context, self.indices, request.get("max_images")):
                raise ValueError("Real output provenance mismatch.")
        self.run = {**context.metadata, "mode": "synthetic", "indices": self.indices, "synthetic_model_seed": 42}
        if self.profile["mode"] == "real":
            self.run.update(mode="real", notice=self.profile["notice"], execution=self.profile["execution"],
                            real_image_evaluation_enabled=True)
            self.run.pop("synthetic_model_seed")
            self.run.pop("synthetic_execution", None)
        self.run_digest = data.digest(canonical(self.run))
        base = context.root / "outputs/xai/phase3b_v1.0.0" / self.profile["namespace"]
        # Short directory names support Windows path limits. Full SHA-256 and
        # exact run bytes remain checked, so prefix collisions fail closed.
        self.directory = base / self.run_digest[:16]
        if self.directory.resolve() != self.directory or not self.directory.is_relative_to(context.root):
            raise PermissionError("Output path must not follow aliases or escape the project.")
        existed = self.directory.exists()
        if existed and not resume:
            raise FileExistsError("Run exists; explicit --resume required.")
        if not existed and resume:
            raise FileNotFoundError("Cannot resume a missing run.")
        if not existed:
            self.directory.mkdir(parents=True, exist_ok=False)
            self._publish("run.json", self.run)
        elif (self.directory / "run.json").read_bytes() != canonical(self.run):
            raise ValueError("Resume run provenance/schema mismatch.")
        allowed = {"run.json", "summary.json"} | {self._name(i) for i in self.indices}
        if any(p.name not in allowed or p.is_symlink() or not p.is_file() for p in self.directory.iterdir()):
            raise ValueError("Partial, unexpected, or aliased result files.")
        for index in self.indices:
            if (self.directory / self._name(index)).exists():
                self.read(index)

    @staticmethod
    def _name(index):
        return f"image_{index:04d}.json"

    def _publish(self, name, value, *, resources=None):
        if name not in {"run.json", "summary.json", *(self._name(i) for i in self.indices)}:
            raise PermissionError("Output name is not part of the run schema.")
        base = self.context.root / "outputs/xai/phase3b_v1.0.0" / self.profile["namespace"]
        if self.directory.resolve() != self.directory or not self.directory.is_relative_to(base):
            raise PermissionError("Output directory drift or protected destination.")
        encoded = canonical(value)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.directory, suffix=".pending", delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            if resources is not None:
                resources.check()  # Last cooperative check AFTER serialization/fsync, BEFORE publication.
            os.link(temporary, self.directory / name)  # Atomic and refuses an existing destination.
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _validate(self, index, payload):
        entry = self.context.authorize(index)
        if (payload.get("schema") != SCHEMA or payload.get("origin") != self.profile["origin"] or
                payload.get("scientific_validation_evaluated") is not self.profile["scientific_validation_evaluated"] or
                payload.get("dataset_index") != index or payload.get("ground_truth_label") != entry.ground_truth_label or
                payload.get("provenance") != primitive(entry) or payload.get("approved_unavailable") != UNAVAILABLE):
            raise ValueError("Image result identity/provenance mismatch.")
        if self.profile["mode"] == "real":
            diagnostics = payload.get("ood_diagnostics", {})
            if (diagnostics.get("status") != "available" or diagnostics.get("classifier_curves_filtered") is not False or
                    not diagnostics.get("calls") or any(type(r.get("accepted")) is not bool for r in diagnostics["calls"])):
                raise ValueError("Frozen OOD diagnostics missing or malformed.")
        record = restore_record(payload)
        if (record.dataset_index != index or record.ground_truth_label != entry.ground_truth_label or
                record.predicted_class != payload.get("predicted_class") or
                payload.get("fixed_target_class") != record.predicted_class):
            raise ValueError("Paired record/target identity mismatch.")
        score = faith.target_scores(payload["logits"], record.predicted_class)
        if (payload.get("confidence") != score.probability or payload.get("classification_correct") is not
                (record.predicted_class == record.ground_truth_label)):
            raise ValueError("Classification metadata mismatch.")
        required = set(UNAVAILABLE) | {stats.CAM, stats.BRIGHTNESS, stats.RANDOM}
        required |= {f"sanity.stage{stage}.{field}" for stage in range(1, 7)
                     for field in ("jaccard", "spearman", "total_variation")}
        required |= {f"{kind}.{policy}.{field}.{ranking}" for kind in ("faithfulness", "auc")
                     for policy in faith.REPLACEMENTS for field in ("probability", "logit", "margin")
                     for ranking in ("gradcam", "brightness", "random")}
        if not required <= set(dict(record.metrics)):
            raise ValueError("Partial scalar metric record.")
        for name, reason in UNAVAILABLE.items():
            if dict(record.metrics).get(name) != stats.observation(reason=reason):
                raise ValueError("Approved unavailability must not disappear or become zero.")
        if (set(payload.get("faithfulness", {})) != set(faith.REPLACEMENTS) or
                len(payload.get("stability", [])) != 30 or len(payload.get("sanity", [])) != 31 or
                len(payload.get("emission", [])) != 3):
            raise ValueError("Partial module result record.")
        expected_pairs = [(None, 0)] + [(seed, stage) for seed in sanity.SEEDS for stage in range(1, 7)]
        if [(r["seed"], r["stage"]) for r in payload["sanity"]] != expected_pairs:
            raise ValueError("Duplicate/incomplete randomization stages.")
        for row in payload["sanity"]:
            if (row["fixed_target_class"] != record.predicted_class or
                    row["modules_randomized"] != list(sanity.STAGES[:row["stage"]]) or
                    not all(row[k] is True for k in ("original_state_preserved", "batchnorm_buffers_preserved",
                                                     "evaluation_mode_preserved", "scoring_state_preserved"))):
                raise ValueError("Randomization state/target drift.")
        if [r["transformation"] for r in payload["stability"]] != primitive(stability.protocol_transformations()):
            raise ValueError("Duplicate/incomplete transformation records.")
        if any(r["target_class"] != record.predicted_class or r["dataset_index"] != index for r in payload["stability"]):
            raise ValueError("Stability target/identity mismatch.")
        counts = {group: sum(r["case_group"] == group for r in payload["stability"])
                  for group in ("prediction_preserving", "class_flipped")}
        if (sum(counts.values()) != 30 or payload.get("stability_case_counts") != counts or
                any(type(r["transformed_predicted_class"]) is not int or r["transformed_predicted_class"] not in (0, 1)
                    for r in payload["stability"])):
            raise ValueError("Stability class/group metadata mismatch.")
        for policy, result in payload["faithfulness"].items():
            expected = [("gradcam", None), ("brightness", None)] + [("random", s) for s in faith.RANDOM_SEEDS]
            if result["status"] == "available":
                rows, primary = result["secondary"], result["primary"]
                if [(r["ranking"], r["seed"]) for r in primary] != expected:
                    raise ValueError("Partial/duplicate primary deletion records.")
            elif result["status"] == "unavailable" and result["reasons"] == ["original_gradcam_degenerate"]:
                rows, primary = result["independent_controls"], None
                expected = expected[1:]
            else:
                raise ValueError("Unknown faithfulness completion status.")
            if [(r["ranking"], r["seed"]) for r in rows] != expected:
                raise ValueError("Partial/duplicate secondary deletion records.")
            for position, row in enumerate(rows):
                raw = row["curve"]
                curve = faith.FullDeletionCurve(raw["target_class"], tuple(raw["original_logits"]),
                                               tuple(tuple(z) for z in raw["logits"]), tuple(raw["fractions"]))
                if curve.target_class != record.predicted_class or tuple(payload["logits"]) != curve.original_logits:
                    raise ValueError("Deletion target/logit mismatch.")
                if primitive(faith.deletion_auc(curve)) != row["auc"]:
                    raise ValueError("Deletion AUC integrity mismatch.")
                six = faith.DeletionCurve(curve.target_class, curve.original_logits, curve.logits[1:7])
                if primary is not None and (primitive(six) != primary[position]["curve"] or
                                             primitive(faith.aopc(six)) != primary[position]["aopc"]):
                    raise ValueError("Primary/secondary pairing mismatch.")

    def write(self, index, payload, *, resources=None):
        if index not in self.indices:
            raise PermissionError("Image is not part of this run.")
        _timed(resources, "result_integrity_validation", self._validate, index, payload)
        self._publish(self._name(index), {"run_sha256": self.run_digest,
                                        "payload_sha256": data.digest(canonical(payload)), "payload": payload},
                      resources=resources)

    def read(self, index):
        if index not in self.indices:
            raise PermissionError("Image is not part of this run.")
        self.context.authorize(index)
        path = self.directory / self._name(index)
        if path.resolve() != path:
            raise PermissionError("Result path alias drift.")
        raw = path.read_bytes()
        envelope = json.loads(raw)
        if (raw != canonical(envelope) or envelope["run_sha256"] != self.run_digest or
                envelope["payload_sha256"] != data.digest(canonical(envelope["payload"]))):
            raise ValueError("Result checksum/canonicalization mismatch.")
        self._validate(index, envelope["payload"])
        return envelope["payload"]

    def finish(self, *, resources=None):
        payloads = [_timed(resources, "result_integrity_validation", self.read, i) for i in self.indices]
        summary = _timed(resources, "statistical_aggregation", summarize_results, payloads, profile=self.profile)
        destination = self.directory / "summary.json"
        if destination.exists():
            if destination.read_bytes() != canonical(summary):
                raise ValueError("Completed summary integrity mismatch.")
        else:
            self._publish("summary.json", summary, resources=resources)
        return summary


def run_synthetic(root=PROJECT_ROOT, indices=None, *, resume=False):
    context = VerifiedContext(root)
    if indices is None:
        indices = sorted(next(e.dataset_index for e in context.access.validation.entries if e.ground_truth_label == c)
                         for c in (0, 1))
    indices = tuple(indices)
    store = ResultStore(context, indices, resume=resume)  # Authorize before constructing any model/array.
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(42)
            model = _SyntheticModel().eval()
        for p in model.parameters():
            p.requires_grad_(False)
        for index in indices:
            if (store.directory / store._name(index)).exists():
                store.read(index)
            else:
                store.write(index, evaluate_synthetic(context, index, model))
        context.check_protected()
        summary = store.finish()
        return {"directory": str(store.directory), "summary": summary}
    finally:
        torch.set_num_threads(previous_threads)


def run_real(root=PROJECT_ROOT, indices=(), *, max_images=None, approval=None, resume=False,
             max_working_set_bytes=2*1024**3, max_image_seconds=900):
    """Attach sanitized diagnostics even when setup/gate/resource initialization fails."""
    started = time.perf_counter()
    try:
        return _run_real(root, indices, max_images=max_images, approval=approval, resume=resume,
                         max_working_set_bytes=max_working_set_bytes, max_image_seconds=max_image_seconds)
    except Exception as error:
        if not hasattr(error, "pilot_telemetry"):
            error.pilot_telemetry = {"completed_images": [], "failed_scope": None,
                                     "failure": _failure_diagnostics(error)}
        error.pilot_telemetry["execution_elapsed_seconds"] = time.perf_counter()-started
        raise


def _run_real(root=PROJECT_ROOT, indices=(), *, max_images=None, approval=None, resume=False,
              max_working_set_bytes=2*1024**3, max_image_seconds=900):
    """Separate approval gate; NEVER called by synthetic or metadata-only modes."""
    from src.xai.execution_adapter import RealExecutionAdapter
    if type(approval) is not dict or approval.get("approved") is not True:
        raise PermissionError("Real evaluation is disabled; explicit separate approval is required.")
    # Fail on unavailable memory measurement BEFORE model loading/pixel access.
    setup_resources = PilotResources(max_working_set_bytes=max_working_set_bytes, max_image_seconds=max_image_seconds)
    context = VerifiedContext(root)
    adapter = RealExecutionAdapter(context, indices, max_images=max_images, approval=approval)
    store = ResultStore(context, adapter.indices, resume=resume, profile=adapter.profile)
    setup_resources.check()
    setup_telemetry = setup_resources.report()
    telemetry = []  # Nondeterministic instrumentation NEVER enters scientific payloads/digests.
    previous_threads = torch.get_num_threads()
    current_resources = None
    try:
        torch.set_num_threads(1)
        for index in adapter.indices:
            if (store.directory / store._name(index)).exists():
                store.read(index)
                continue
            current_resources = PilotResources(max_working_set_bytes=max_working_set_bytes,
                                               max_image_seconds=max_image_seconds)

            def execute():
                entry = context.authorize(index)
                current_resources.check()
                model = adapter.model()
                current_resources.check()
                image = adapter.image(index)
                current_resources.check()
                score, diagnostics = adapter.diagnostic_scorer(model)
                score = current_resources.ood_scorer(score)
                return _evaluate_image(context, index, model, image, entry, profile=adapter.profile,
                                       scorer=score, diagnostics=diagnostics, resources=current_resources)

            payload = execute()
            current_resources.check()
            store.write(index, payload, resources=current_resources)
            current_resources.check()
            telemetry.append({"dataset_index": index, "status": "completed", **current_resources.report()})
            current_resources = None
        aggregation_resources = PilotResources(max_working_set_bytes=max_working_set_bytes,
                                               max_image_seconds=max_image_seconds)
        current_resources = aggregation_resources
        summary = store.finish(resources=aggregation_resources)
        context.check_protected()
        # Caller may capture stdout to a separate operational log. No telemetry
        # is embedded into atomic scientific records or their resume identity.
        return {"directory": str(store.directory), "summary": summary, "telemetry": telemetry,
                "aggregation_telemetry": aggregation_resources.report(),
                "setup_telemetry": setup_telemetry}
    except Exception as error:
        # Failed/partial images never reach ResultStore.write. Preserve operational
        # diagnostics on the exception; no fabricated successful result is returned.
        error.pilot_telemetry = {"completed_images": telemetry, "failed_scope":
                                 current_resources.report() if current_resources is not None else None,
                                 "failure": _failure_diagnostics(error, current_resources)}
        raise
    finally:
        torch.set_num_threads(previous_threads)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("synthetic", "dry-run", "real"))
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--index", type=int, action="append", dest="indices")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--approval-file", type=Path)
    parser.add_argument("--max-working-set-bytes", type=int, default=2*1024**3)
    parser.add_argument("--max-image-seconds", type=float, default=900)
    args = parser.parse_args(argv)
    if args.mode != "real" and (args.max_images is not None or args.approval_file is not None or
                                args.max_working_set_bytes != 2*1024**3 or args.max_image_seconds != 900):
        parser.error("Execution approval and bounds apply only to the separate real mode.")
    if args.mode == "dry-run":
        if args.resume or args.indices:
            parser.error("Metadata dry-run always covers the full validation manifest and writes no results.")
        result = dry_run(args.project_root)
    elif args.mode == "synthetic":
        result = run_synthetic(args.project_root, args.indices, resume=args.resume)
    else:
        if args.approval_file is None or not args.indices or args.max_images is None:
            parser.error("Real mode is disabled without approval file, explicit validation indices and image bound.")
        cli_started = time.perf_counter()
        try:
            result = run_real(args.project_root, args.indices, max_images=args.max_images,
                              approval=json.loads(args.approval_file.read_bytes()), resume=args.resume,
                              max_working_set_bytes=args.max_working_set_bytes, max_image_seconds=args.max_image_seconds)
        except Exception as error:
            # Operational JSON goes to stderr; stdout remains reserved for a
            # completed run. Suppress unsanitized traceback/messages at the CLI.
            failure = {"schema": "Phase3B-operational-failure-v1.0.0", "status": "failed",
                       "scientific_completion_reported": False,
                       "telemetry": getattr(error, "pilot_telemetry", {
                           "execution_elapsed_seconds": time.perf_counter()-cli_started,
                           "failure": _failure_diagnostics(error)})}
            print(canonical(failure).decode("utf-8"), file=sys.stderr, end="")
            raise SystemExit(1) from None
    print(canonical(result).decode("utf-8"), end="")


if __name__ == "__main__":
    main()
