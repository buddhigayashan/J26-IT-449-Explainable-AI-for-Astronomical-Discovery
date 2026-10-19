"""Frozen Phase3B-XAI-v1.0.0 statistics on supplied scalar records only.

Validation is exploratory; candidate-source independence is not verified.
Paired sign flips require appropriate paired-null symmetry/exchangeability.
Intervals condition on metric availability, not retrained-model or survey
variation. Repeats, pixels and transformations are NEVER independent units.
No dataset/model/image loading, global RNG changes, or artifact writes occur.

Conventions: linear-interpolated quartiles/percentiles; descriptive sample SD
(ddof=1, unavailable for n<2). Descriptive repeat means use available seeds and
retain coverage; the primary random control requires ALL 20 seeds. An absent
class blocks inference. Singleton class strata are flagged but not silently
excluded: the frozen rule requires two images overall, not two per class.
"""

from dataclasses import dataclass
from numbers import Real
from typing import Mapping

import numpy as np

REPLICATES = 10000
BOOTSTRAP_SEED = 42
PERMUTATION_SEED = 43
ALPHA = 0.05
PRIMARY_FAMILY = ("CAM_minus_random_probability_AOPC30",
                  "CAM_minus_brightness_probability_AOPC30")
CAM = "cam_probability_aopc30"
RANDOM = "random_probability_aopc30"
BRIGHTNESS = "brightness_probability_aopc30"


@dataclass(frozen=True)
class Observation:
    value: float | None
    status: str
    reasons: tuple[str, ...] = ()
    repeat_kind: str | None = None
    setting: float | int | None = None
    repeats: tuple[tuple[int, "Observation"], ...] = ()

    @property
    def available_repeats(self):
        return sum(value.status == "available" for _, value in self.repeats)


def observation(value=None, *, reason=None):
    """Missing/nonfinite values become unavailable, never numerical zero."""
    if reason is not None:
        if not isinstance(reason, str) or not reason:
            raise ValueError("Unavailable reason must be a nonempty string.")
        return Observation(None, "unavailable", (reason,))
    if value is None:
        return Observation(None, "unavailable", ("missing",))
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("Metrics must be scalar real values, not pixels/arrays.")
    if not np.isfinite(value):
        return Observation(None, "unavailable", ("nonfinite",))
    return Observation(float(value), "available")


def _checked_observation(value):
    if not isinstance(value, Observation):
        return observation(value)
    if (type(value.reasons) is not tuple or
            any(not isinstance(r, str) or not r for r in value.reasons)):
        raise ValueError("Invalid observation reasons.")
    if value.status == "available":
        if observation(value.value).status != "available" or value.reasons:
            raise ValueError("Available observations require a finite scalar and no failure reasons.")
    elif value.status != "unavailable" or value.value is not None or not value.reasons:
        raise ValueError("Unavailable observations require explicit reasons and value=None.")
    if value.repeat_kind is None:
        if value.repeats or value.setting is not None:
            raise ValueError("Repeats require a declared family and setting.")
    else:
        # Reconstruct to reject forged aggregate values/coverage.
        expected = aggregate_repeats(dict(value.repeats), value.repeat_kind, value.setting)
        if expected != value:
            raise ValueError("Repeat aggregate is inconsistent with its raw measurements.")
    return value


def aggregate_repeats(values: Mapping, kind: str, setting=None):
    """One image/metric/setting ONLY. Keep seeds attached, do not pool settings.

    Missing expected seeds are retained as unavailable. Extra/duplicate seeds,
    nested aggregates and unsupported settings are configuration errors.
    """
    if kind == "random_deletion" and setting is None:
        seeds = tuple(range(1000, 1020))
    elif kind == "noise" and type(setting) in (int, float) and setting in (.005, .01):
        seeds = tuple(range(2000, 2010))
    elif kind == "randomization" and type(setting) is int and setting in range(1, 7):
        seeds = tuple(range(3000, 3005))
    else:
        raise ValueError("Unknown repeat family/setting.")
    if not isinstance(values, Mapping) or any(type(k) is not int or k not in seeds for k in values):
        raise ValueError("Unexpected repeat seed.")
    raw = []
    for seed in seeds:
        item = _checked_observation(values[seed]) if seed in values else observation(reason="missing_repeat")
        if item.repeat_kind is not None:
            raise ValueError("Nested repeat aggregates are forbidden.")
        raw.append((seed, item))
    valid = [item.value for _, item in raw if item.status == "available"]
    if not valid:
        result = observation(reason="all_repeats_unavailable")
    elif kind == "random_deletion" and len(valid) != len(seeds):
        result = observation(reason="incomplete_random_controls")
    else:
        result = observation(float(np.mean(valid)))
    return Observation(result.value, result.status, result.reasons, kind, setting, tuple(raw))


@dataclass(frozen=True)
class ImageRecord:
    dataset_index: int
    ground_truth_label: int
    predicted_class: int | None
    metrics: tuple[tuple[str, Observation], ...]


def image_record(dataset_index, ground_truth_label, predicted_class, metrics: Mapping):
    """Construct an immutable scalar record; upstream provenance remains required."""
    if not isinstance(metrics, Mapping) or any(not isinstance(k, str) or not k for k in metrics):
        raise ValueError("Metrics must have nonempty names.")
    result = ImageRecord(dataset_index, ground_truth_label, predicted_class,
                         tuple(sorted((k, _checked_observation(v)) for k, v in metrics.items())))
    _records((result,))
    return result


def _records(records):
    records = tuple(records)
    seen = set()
    for record in records:
        if not isinstance(record, ImageRecord):
            raise ValueError("Require image records, not repeated-measurement rows.")
        if type(record.dataset_index) is not int or not 0 <= record.dataset_index < 792:
            raise ValueError("Invalid image identifier.")
        if record.dataset_index in seen:
            raise ValueError("Duplicate image identifier.")
        seen.add(record.dataset_index)
        if type(record.ground_truth_label) is not int or record.ground_truth_label not in (0, 1):
            raise ValueError("Invalid ground-truth class label.")
        if record.predicted_class is not None and (type(record.predicted_class) is not int or
                                                   record.predicted_class not in (0, 1)):
            raise ValueError("Invalid predicted class label.")
        if type(record.metrics) is not tuple:
            raise ValueError("Metrics must be an immutable tuple.")
        names = set()
        for entry in record.metrics:
            if type(entry) is not tuple or len(entry) != 2:
                raise ValueError("Metric entries must be immutable name/observation pairs.")
            name, value = entry
            if not isinstance(name, str) or not name or name in names:
                raise ValueError("Invalid/duplicate metric name.")
            names.add(name)
            if not isinstance(value, Observation):
                raise ValueError("Use image_record() to normalize raw scalar metrics.")
            _checked_observation(value)
    return tuple(sorted(records, key=lambda r: r.dataset_index))


def _metric(record, name):
    return dict(record.metrics).get(name, observation(reason="missing_metric"))


def _reason_counts(observations):
    counts = {}
    for item in observations:
        for reason in set(item.reasons):
            counts[reason] = counts.get(reason, 0) + 1
    return tuple(sorted(counts.items()))


@dataclass(frozen=True)
class Descriptive:
    """Reason counts count affected observations; several reasons may co-occur."""
    total: int
    available: int
    unavailable: int
    unavailable_reasons: tuple[tuple[str, int], ...]
    mean: float | None
    median: float | None
    q1: float | None
    q3: float | None
    iqr: float | None
    sample_standard_deviation: float | None
    status: str
    warnings: tuple[str, ...]
    repeat_total: int
    repeat_available: int
    repeat_unavailable_reasons: tuple[tuple[str, int], ...]

    @property
    def repeat_unavailable(self):
        return self.repeat_total - self.repeat_available


def _describe(observations):
    observations = tuple(observations)
    values = np.array([v.value for v in observations if v.status == "available"], dtype=np.float64)
    n = len(values)
    repeats = [v for item in observations for _, v in item.repeats]
    if n:
        q1, median, q3 = np.percentile(values, [25, 50, 75], method="linear")
        quantities = (float(values.mean()), float(median), float(q1), float(q3), float(q3-q1))
    else:
        quantities = (None,) * 5
    return Descriptive(len(observations), n, len(observations)-n,
                       _reason_counts(observations), *quantities,
                       float(values.std(ddof=1)) if n >= 2 else None,
                       "available" if n else "unavailable",
                       ("fewer_than_two_available_images",) if n < 2 else (),
                       len(repeats), sum(v.status == "available" for v in repeats),
                       _reason_counts(repeats))


@dataclass(frozen=True)
class EndpointSummary:
    metric: str
    overall: Descriptive
    by_class: tuple[Descriptive, Descriptive]  # FRI, FRII
    correctly_classified: Descriptive
    misclassified: Descriptive
    unknown_prediction: Descriptive
    sample_weighted_overall_mean: float | None
    equal_class_macro_mean: float | None
    macro_status: str


def summarize(records, metric):
    """Metric-specific coverage; strata are descriptive and never new tests."""
    records = _records(records)
    groups = tuple(_describe(_metric(r, metric) for r in records if r.ground_truth_label == label)
                   for label in (0, 1))
    overall = _describe(_metric(r, metric) for r in records)
    macro = None if any(g.mean is None for g in groups) else (groups[0].mean + groups[1].mean) / 2
    return EndpointSummary(metric, overall, groups,
                           _describe(_metric(r, metric) for r in records if
                                     r.predicted_class == r.ground_truth_label),
                           _describe(_metric(r, metric) for r in records if
                                     r.predicted_class is not None and r.predicted_class != r.ground_truth_label),
                           _describe(_metric(r, metric) for r in records if r.predicted_class is None),
                           overall.mean, macro, "available" if macro is not None else "missing_class")


def add_primary_contrasts(records):
    """Primary border-median probability AOPC30 inputs only (caller provenance).

    Each affected pair is complete-case independently. Random control means must
    retain their verified 20 seeded measurements; a bare scalar is insufficient.
    Missing deletion steps must already be represented as unavailable upstream.
    """
    output = []
    for record in _records(records):
        metrics = dict(record.metrics)
        if any(name in metrics for name in PRIMARY_FAMILY):
            raise ValueError("Refusing to overwrite primary contrasts.")
        cam = _metric(record, CAM)
        for endpoint, baseline in zip(PRIMARY_FAMILY, (RANDOM, BRIGHTNESS)):
            control = _metric(record, baseline)
            reasons = tuple(f"{name}:{reason}" for name, item in ((CAM, cam), (baseline, control))
                            for reason in item.reasons)
            if baseline == RANDOM and control.repeat_kind != "random_deletion":
                reasons += ("unverified_random_control_aggregation",)
            metrics[endpoint] = (Observation(None, "unavailable", reasons) if reasons else
                                 observation(cam.value-control.value))
        output.append(image_record(record.dataset_index, record.ground_truth_label,
                                   record.predicted_class, metrics))
    return tuple(output)


@dataclass(frozen=True)
class Coverage:
    image_ids: tuple[int, ...]
    class_counts: tuple[int, int]
    excluded: tuple[tuple[int, tuple[str, ...]], ...]
    conditional_on_metric_availability: bool
    status: str
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]


def _complete_cases(records, endpoints):
    records = _records(records)
    endpoints = tuple(endpoints)
    if not endpoints or any(not isinstance(e, str) or not e for e in endpoints) or len(set(endpoints)) != len(endpoints):
        raise ValueError("Require distinct nonempty endpoint names.")
    valid, excluded = [], []
    for record in records:
        reasons = tuple(f"{e}:{reason}" for e in endpoints for reason in _metric(record, e).reasons)
        (excluded if reasons else valid).append((record.dataset_index, reasons) if reasons else record)
    counts = tuple(sum(r.ground_truth_label == c for r in valid) for c in (0, 1))
    reasons = tuple(r for r, condition in (("missing_class", 0 in counts),
                                          ("fewer_than_two_images", len(valid) < 2)) if condition)
    coverage = Coverage(tuple(r.dataset_index for r in valid), counts, tuple(excluded),
                        bool(excluded), "unavailable" if reasons else "available", reasons,
                        ("singleton_class_stratum",) if 1 in counts else ())
    return tuple(valid), endpoints, coverage


def _bootstrap_indices(labels):
    """Each row is one paired image resample; no repeat-level resampling."""
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    strata = [np.flatnonzero(labels == c) for c in (0, 1)]
    return np.concatenate([rng.choice(group, size=(REPLICATES, len(group)), replace=True)
                           for group in strata], axis=1)


@dataclass(frozen=True)
class BootstrapResult:
    endpoints: tuple[str, ...]
    coverage: Coverage
    macro_estimates: tuple[float, ...] | None
    overall_estimates: tuple[float, ...] | None
    macro_intervals: tuple[tuple[float, float], ...] | None
    overall_intervals: tuple[tuple[float, float], ...] | None
    macro_replicates: tuple[tuple[float, ...], ...]
    overall_replicates: tuple[tuple[float, ...], ...]
    replicates: int = REPLICATES
    seed: int = BOOTSTRAP_SEED
    interval_percentiles: tuple[float, float] = (2.5, 97.5)


def bootstrap_paired(records, endpoints):
    """Paired percentile intervals for jointly complete requested endpoints.

    Both sample-weighted and equal-class macro means share image resamples.
    For metric-specific denominators call separately per endpoint/paired contrast;
    joint calls intentionally condition on availability of ALL requested values.
    Raw repeats remain attached to their source image and are not resampled.
    Returned intervals are pointwise, not simultaneous or hypothesis tests.
    """
    records, endpoints, coverage = _complete_cases(records, endpoints)
    if coverage.status != "available":
        return BootstrapResult(endpoints, coverage, None, None, None, None, (), ())
    labels = np.array([r.ground_truth_label for r in records])
    table = np.array([[_metric(r, e).value for e in endpoints] for r in records])
    sampled = table[_bootstrap_indices(labels)]
    n0 = coverage.class_counts[0]
    macro = (sampled[:, :n0].mean(axis=1) + sampled[:, n0:].mean(axis=1)) / 2
    overall = sampled.mean(axis=1)
    intervals = lambda x: tuple(map(tuple, np.percentile(x, [2.5, 97.5], axis=0, method="linear").T.tolist()))
    macro_estimate = (table[labels == 0].mean(axis=0) + table[labels == 1].mean(axis=0)) / 2
    return BootstrapResult(endpoints, coverage, tuple(macro_estimate.tolist()), tuple(table.mean(axis=0).tolist()),
                           intervals(macro), intervals(overall), tuple(map(tuple, macro.tolist())),
                           tuple(map(tuple, overall.tolist())))


@dataclass(frozen=True)
class SignFlipResult:
    endpoint: str
    coverage: Coverage
    statistic: float | None
    as_or_more_extreme: int | None
    p_value: float | None
    permutations: int = REPLICATES
    seed: int = PERMUTATION_SEED


def primary_sign_flip(records, endpoint):
    """Only the two frozen primary endpoints; two-sided plus-one image test."""
    if endpoint not in PRIMARY_FAMILY:
        raise ValueError("No additional confirmatory hypothesis tests are authorized.")
    records, _, coverage = _complete_cases(records, (endpoint,))
    if coverage.status != "available":
        return SignFlipResult(endpoint, coverage, None, None, None)
    values = np.array([_metric(r, endpoint).value for r in records])
    labels = np.array([r.ground_truth_label for r in records])
    weights = np.array([1 / (2 * coverage.class_counts[c]) for c in labels])
    weighted = values * weights
    statistic = float(weighted.sum())
    rng = np.random.Generator(np.random.PCG64(PERMUTATION_SEED))
    signs = 2 * rng.integers(0, 2, size=(REPLICATES, len(records)), dtype=np.int8) - 1
    permuted = (signs * weighted).sum(axis=1)
    extreme = int(np.count_nonzero(np.abs(permuted) >= abs(statistic)))
    return SignFlipResult(endpoint, coverage, statistic, extreme, (1+extreme)/(1+REPLICATES))


@dataclass(frozen=True)
class HolmResult:
    endpoint: str
    raw_p_value: float | None
    adjusted_p_value: float | None
    reject_at_005: bool | None


def holm_primary(p_values: Mapping):
    """Exactly two hypotheses at alpha=.05; unavailable tests do not shrink m."""
    if set(p_values) != set(PRIMARY_FAMILY):
        raise ValueError("Holm family must contain exactly the two frozen primary tests.")
    for value in p_values.values():
        if value is not None and (isinstance(value, (bool, np.bool_)) or
                                  not isinstance(value, Real) or not np.isfinite(value) or not 0 <= value <= 1):
            raise ValueError("Invalid p-value.")
    ordered = sorted(PRIMARY_FAMILY, key=lambda e: (1 if p_values[e] is None else p_values[e],
                                                   PRIMARY_FAMILY.index(e)))
    adjusted, previous = {}, 0.
    for rank, endpoint in enumerate(ordered):
        value = p_values[endpoint]
        previous = max(previous, min(1., (2-rank)*(1 if value is None else value)))
        adjusted[endpoint] = None if value is None else previous
    return tuple(HolmResult(e, p_values[e], adjusted[e],
                            None if adjusted[e] is None else adjusted[e] <= ALPHA) for e in PRIMARY_FAMILY)


@dataclass(frozen=True)
class PrimaryAnalysis:
    records: tuple[ImageRecord, ...]
    summaries: tuple[EndpointSummary, ...]
    bootstrap: tuple[BootstrapResult, ...]
    tests: tuple[SignFlipResult, ...]
    holm: tuple[HolmResult, ...]


def analyze_primary(records):
    """Analyze supplied primary AOPCs only; executes no image evaluation."""
    derived = add_primary_contrasts(records)
    tests = tuple(primary_sign_flip(derived, e) for e in PRIMARY_FAMILY)
    return PrimaryAnalysis(derived, tuple(summarize(derived, e) for e in PRIMARY_FAMILY),
                           tuple(bootstrap_paired(derived, (e,)) for e in PRIMARY_FAMILY), tests,
                           holm_primary({t.endpoint: t.p_value for t in tests}))
