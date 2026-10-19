"""Phase 3B metadata-only provenance, immutable manifests and guarded dispatch.

No pickle is executed, no NumPy image array is reconstructed, and no files are
written here. Archive members are inspected as bytes/opcodes only. The guard
protects this interface; it cannot prevent callers from using unrelated loaders.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import pickletools
import re
import tarfile
from typing import Callable, Literal

import yaml

PROTOCOL = "Phase3B-XAI-v1.0.0"
CONFIG_PATH = "config/xai_phase3b_v1.0.0.yaml"
# SHA-256 of canonical parsed YAML, pinned independently of the supplied file.
CONFIG_SHA256 = "45fb3e86b681c6f73035a7743fa0223eb271445ee6ea06bf9a2604551ef3ad84"
MEMBERS = tuple(f"batches/data_batch_{i}" for i in range(1, 9)) + ("batches/test_batch",)
FILENAME = re.compile(r"BestCat_proc/Cropped/\d+_([0-9]+(?:\.[0-9]+)?)_([+-][0-9]+(?:\.[0-9]+)?)\.png")


def canonical_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def digest(data: bytes, algorithm="sha256") -> str:
    return hashlib.new(algorithm, data).hexdigest()


def candidate_source_key(filename: str) -> str:
    if not isinstance(filename, str) or not (match := FILENAME.fullmatch(filename)):
        raise ValueError("Malformed filename provenance; expected frozen MiraBest filename pattern.")
    return "filename-coordinate-fields:" + "|".join(match.groups())


@dataclass(frozen=True)
class MetadataRow:
    dataset_index: int
    ground_truth_label: int
    original_filename: str
    candidate_source_key: str
    archive_member: str
    member_row: int


@dataclass(frozen=True)
class Provenance:
    protocol_version: str
    protocol_configuration_sha256: str
    checkpoint_reference: str
    checkpoint_sha256: str
    frozen_split_reference: str
    frozen_split_sha256: str
    dataset_reference: str
    dataset_sha256: str
    dataset_md5: str
    source_reference: str
    metadata_sha256: str


@dataclass(frozen=True)
class SelectionEntry:
    protocol_version: str
    dataset_index: int
    split_membership: str
    ground_truth_label: int
    original_filename: str
    candidate_source_key: str
    selection_rule: str
    frozen_split_hash: str
    dataset_provenance: Provenance
    archive_member: str
    member_row: int


@dataclass(frozen=True)
class SelectionManifest:
    cohort: str
    entries: tuple[SelectionEntry, ...]
    provenance: Provenance

    def to_bytes(self) -> bytes:
        """Canonical immutable-in-content representation, without timestamps or writes."""
        return canonical_bytes(asdict(self))

    @property
    def sha256(self):
        return digest(self.to_bytes())


def _member_metadata(raw: bytes, member: str, start: int) -> tuple[MetadataRow, ...]:
    """Extract simple labels/filename scalars; never execute GLOBAL/REDUCE opcodes.

    Supports the frozen Python-2 NumPy Unicode filename scalars and ordinary
    string filename lists. Image payload arguments are ignored, not reconstructed.
    """
    labels, filenames = [], []
    in_labels = in_filenames = False
    seen_labels = seen_filenames = False
    last = None
    try:
        for op, arg, position in pickletools.genops(raw):
            last = (op.name, position)
            small = arg if isinstance(arg, (str, bytes)) and len(arg) <= 512 else None
            key = small.decode("ascii", errors="ignore") if isinstance(small, bytes) else small
            if key == "labels":
                if seen_labels:
                    raise ValueError("Duplicate labels metadata.")
                seen_labels = in_labels = True
                continue
            if key == "data":
                in_labels = False
                continue
            if key == "filenames":
                in_labels = False
                seen_filenames = in_filenames = True
                continue
            if in_labels:
                if op.name in {"INT", "BININT", "BININT1", "BININT2", "LONG", "LONG1", "LONG4"}:
                    if type(arg) is not int or arg not in (0, 1):
                        raise ValueError("Unsupported ground-truth label.")
                    labels.append(arg)
                elif op.name not in {"EMPTY_LIST", "MARK", "BINPUT", "LONG_BINPUT", "PUT",
                                     "APPEND", "APPENDS", "MEMOIZE"}:
                    raise ValueError("Malformed labels metadata.")
            if in_filenames and small is not None:
                encoded = small.encode("latin1") if isinstance(small, str) else small
                try:
                    text = encoded.decode("utf-32-le") if b"\x00" in encoded else encoded.decode("ascii")
                except UnicodeError:
                    continue
                if text.endswith(".png"):
                    candidate_source_key(text)  # Reject malformed paths, do not silently omit.
                    filenames.append(text)
        if last != ("STOP", len(raw) - 1):
            raise ValueError("Incomplete or trailing pickle metadata.")
    except (ValueError, UnicodeError, OverflowError) as error:
        raise ValueError(f"Malformed metadata in {member}: {error}") from error
    if not seen_labels or not seen_filenames or len(labels) != 88 or len(filenames) != 88:
        raise ValueError(f"Missing/malformed metadata in {member}; expected 88 labels and filenames.")
    return tuple(MetadataRow(start + i, label, name, candidate_source_key(name), member, i)
                 for i, (label, name) in enumerate(zip(labels, filenames)))


def _archive_metadata(archive_bytes: bytes) -> tuple[MetadataRow, ...]:
    import io
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive member names.")
        required = set(MEMBERS) | {"batches/hash.md5", "batches/batches.meta"}
        files = {m.name for m in members if m.isfile()}
        if files != required or any(m.issym() or m.islnk() for m in members):
            raise ValueError("Dataset archive layout mismatch.")
        manifest = {}
        for line in archive.extractfile("batches/hash.md5").read().decode("ascii").splitlines():
            checksum, name = line.split()
            full = "batches/" + name
            if full in manifest or not re.fullmatch(r"[0-9a-f]{32}", checksum):
                raise ValueError("Malformed dataset checksum manifest.")
            manifest[full] = checksum
        if set(manifest) != required - {"batches/hash.md5"}:
            raise ValueError("Incomplete dataset checksum manifest.")
        payloads = {}
        for name, expected in manifest.items():
            raw = archive.extractfile(name).read()
            if digest(raw, "md5") != expected:
                raise ValueError(f"Dataset member checksum mismatch: {name}")
            payloads[name] = raw
        return tuple(row for i, name in enumerate(MEMBERS)
                     for row in _member_metadata(payloads[name], name, 88 * i))


def _validate_splits(splits, counts):
    if not isinstance(splits, dict) or set(splits) != {"train", "validation", "test"}:
        raise ValueError("Saved splits must contain exactly train, validation and test.")
    flattened = []
    for name, expected in counts.items():
        values = splits[name]
        if (not isinstance(values, list) or len(values) != expected
                or any(type(i) is not int or not 0 <= i < 792 for i in values)):
            raise ValueError(f"Invalid saved {name} indices/count.")
        flattened.extend(values)
    if len(flattened) != 792 or set(flattened) != set(range(792)):
        raise ValueError("Saved splits overlap or fail complete disjoint coverage.")


def _validated_inputs(root: Path, config_path: Path):
    config_raw = config_path.read_bytes()
    config = yaml.safe_load(config_raw)
    if digest(canonical_bytes(config)) != CONFIG_SHA256:
        raise ValueError("Frozen protocol configuration hash mismatch.")
    provenance = config["provenance"]
    content = {}
    for key in ("checkpoint", "split", "archive"):
        path = (root / provenance[key]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Provenance reference escapes project root.")
        raw = path.read_bytes()
        if digest(raw) != provenance[key + "_sha256"]:
            raise ValueError(f"Frozen {key} hash mismatch.")
        content[key] = raw
    if digest(content["archive"], "md5") != provenance["archive_md5"]:
        raise ValueError("Frozen archive MD5 mismatch.")
    splits = json.loads(content["split"])
    _validate_splits(splits, config["splits"]["counts"])
    # All subsequent parsing uses these exact checked bytes, not a second read.
    return config, content, splits


@dataclass(frozen=True)
class EvaluationData:
    """Immutable manifests plus a revalidating allowlist boundary for later consumers.

    No decoder/scorer/visualizer is supplied or executed by manifest generation.
    Consumers must call dispatch before their operation, never expose an underlying
    dataset directly. Python callers deliberately bypassing this API are out of scope.
    """

    engineering: SelectionManifest
    validation: SelectionManifest
    _root: Path
    _config_path: Path
    _test_indices: frozenset[int]

    @property
    def validation_allowlist(self):
        return frozenset(entry.dataset_index for entry in self.validation.entries)

    def authorize(self, dataset_index: int, cohort: Literal["engineering", "validation"]) -> SelectionEntry:
        if type(dataset_index) is not int:
            raise PermissionError("Dataset index must be an integer.")
        if dataset_index in self._test_indices:
            raise PermissionError("Locked TEST image access is forbidden in Phase 3B.")
        if cohort not in {"engineering", "validation"}:
            raise PermissionError("Only engineering TRAIN or validation access is permitted.")
        entries = getattr(self, cohort).entries
        entry = next((row for row in entries if row.dataset_index == dataset_index), None)
        if entry is None:
            raise PermissionError("Index is outside the requested frozen cohort allowlist.")
        _validated_inputs(self._root, self._config_path)  # Detect drift even after manifest creation.
        return entry

    def dispatch(self, dataset_index: int, cohort: str, operation: str, handler: Callable):
        """Authorize before any retrieval, decoding, scoring or visualization callback."""
        if operation not in {"retrieve", "decode", "score", "visualize"}:
            raise PermissionError("Unsupported Phase 3B operation.")
        entry = self.authorize(dataset_index, cohort)
        return handler(entry)


def load_evaluation_data(project_root: str | Path, config_path: str | Path | None = None) -> EvaluationData:
    """Validate trusted pins, inspect metadata, and return manifests; never write/decode."""
    root = Path(project_root).resolve()
    path = Path(config_path).resolve() if config_path is not None else root / CONFIG_PATH
    config, content, splits = _validated_inputs(root, path)
    rows = _archive_metadata(content["archive"])
    if len(rows) != 792:
        raise ValueError("Dataset metadata count mismatch.")
    p = config["provenance"]
    provenance = Provenance(PROTOCOL, CONFIG_SHA256, p["checkpoint"], p["checkpoint_sha256"],
                            p["split"], p["split_sha256"], p["archive"], p["archive_sha256"],
                            p["archive_md5"], p["source"], digest(canonical_bytes([asdict(r) for r in rows])))

    def manifest(cohort, indices, split, rule):
        return SelectionManifest(cohort, tuple(
            SelectionEntry(PROTOCOL, i, split, rows[i].ground_truth_label,
                           rows[i].original_filename, rows[i].candidate_source_key,
                           rule, provenance.frozen_split_sha256, provenance,
                           rows[i].archive_member, rows[i].member_row)
            for i in indices), provenance)

    selected = []
    for label in (0, 1):
        candidates = [i for i in splits["train"] if rows[i].ground_truth_label == label]
        ranked = sorted(candidates, key=lambda i: (digest(f"{PROTOCOL}|engineering|42|{i}".encode("ascii")), i))
        if len(ranked) < 20:
            raise ValueError("Insufficient TRAIN metadata for exactly 20 examples per class.")
        selected.extend(ranked[:20])
    engineering = manifest("engineering", selected, "train",
                           config["sampling"]["engineering_key"] + "; first 20 per class by ascending SHA-256, then index")
    validation = manifest("validation", sorted(splits["validation"]), "validation",
                          config["sampling"]["validation"])
    return EvaluationData(engineering, validation, root, path, frozenset(splits["test"]))
