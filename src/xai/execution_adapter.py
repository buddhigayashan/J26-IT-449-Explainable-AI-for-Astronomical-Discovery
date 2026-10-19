"""Non-executable, format-specific MiraBest row access and gated execution.

This is NOT a general pickle loader. The layout reader models a small frozen
protocol-3 grammar symbolically. GLOBAL/REDUCE/BUILD never invoke constructors.
Binary arguments are byte-range descriptors, not NumPy objects or image arrays.
Only a separately authorized row may subsequently become a 150x150 array.
"""

from dataclasses import dataclass
import re
import struct
import io
import json
from pathlib import Path
import tarfile
import time

import numpy as np
import torch

from src.xai import evaluation_data as data
from src.xai import faithfulness as faith
from src.data import preprocess
from src.models import resnet_cnn


ADAPTER_SCHEMA = "Phase3B-execution-adapter-v1.0.0"
REAL_NOTICE = "Exploratory validation evaluation; incomplete frozen secondary coverage."


def preprocessing_provenance():
    """Bind trusted preprocessing and model code; no public guard is changed."""
    return {"image_shape": [150, 150], "grayscale": True, "dtype": "float32",
            "mean": 0.5, "std": 0.5, "augmentation": False,
            "source_sha256": data.digest(Path(preprocess.__file__).read_bytes()),
            "model_source_sha256": data.digest(Path(resnet_cnn.__file__).read_bytes()),
            "adapter_sha256": data.digest(Path(__file__).read_bytes())}


def approval_request(context, indices, max_images):
    """Describe the exact bounded execution to approve; this grants no permission.

    A future explicit approval file must contain this object plus approved=true.
    This is a local operator gate, not cryptographic user authentication.
    """
    indices = tuple(indices)
    if (type(max_images) is not int or not 1 <= max_images <= 119 or not indices or
            len(indices) > max_images or any(type(i) is not int for i in indices)):
        raise ValueError("A mandatory image-count bound in [1,119] must cover the explicit indices.")
    if indices != tuple(sorted(set(indices))):
        raise ValueError("Duplicate or unsorted validation indices.")
    if len(context.access.validation_allowlist) != 119:
        raise ValueError("Exactly 119 frozen validation indices required.")
    for index in indices:
        context.authorize(index)  # BEFORE opening any image payload or model.
    return {"schema": ADAPTER_SCHEMA, "indices": list(indices), "max_images": max_images,
            "context_sha256": data.digest(data.canonical_bytes(context.metadata)),
            "preprocessing": preprocessing_provenance(), "device": "cpu", "torch_threads": 1,
            "class_mapping": {"FR-I": 0, "FR-II": 1}}


def validate_approval(context, indices, max_images, approval=None):
    if type(approval) is not dict or approval.get("approved") is not True:
        raise PermissionError("Real evaluation is disabled; separate explicit approval is required.")
    request = approval_request(context, indices, max_images)
    if approval != {**request, "approved": True}:
        raise ValueError("Execution approval/provenance/bounds mismatch.")
    return request


def decode_authorized_row(stream, size, row, *, expected_label, expected_filename):
    """Internal row decoder AFTER allowlist authorization, not an access grant.

    Only the selected image byte range is read into an array. All other image
    byte arguments are skipped by inspect_member_layout. No pickle executes.
    """
    if type(row) is not int or not 0 <= row < 88:
        raise PermissionError("Invalid authorized row.")
    stream.seek(0)
    layout = inspect_member_layout(stream, size)
    filename = layout.filenames[row]
    if isinstance(filename, _Scalar):
        stream.seek(filename.data.offset)
        filename = stream.read(filename.data.length).decode("utf-32-le")
    if layout.labels[row] != expected_label or filename != expected_filename:
        raise ValueError("Authorized row metadata/provenance mismatch.")
    span = layout.rows[row]
    stream.seek(span.offset)
    raw = stream.read(span.length)
    if len(raw) != 22500:
        raise ValueError("Truncated authorized row.")
    return np.frombuffer(raw, dtype=np.uint8).reshape(150, 150).copy()


class RealExecutionAdapter:
    """Gated, bounded, private CPU execution. Construction performs NO image decode.

    Compressed/serialized bytes may be hashed/copied in full for provenance.
    Those opaque bytes never become a mixed-split NumPy/pickle image object.
    Only image(index), after reauthorization, materializes one allowed row.
    """

    def __init__(self, context, indices, *, max_images=None, approval=None):
        self.context = context
        self.indices = tuple(indices)
        self.request = validate_approval(context, self.indices, max_images, approval)
        self._approval = json.loads(json.dumps(approval))
        self._max_images = max_images
        self._model = None
        self._detector = None
        self._revalidate()
        p = context.access.validation.provenance
        self._archive = self._checked_bytes(p.dataset_reference, p.dataset_sha256)
        self._checkpoint = self._checked_bytes(p.checkpoint_reference, p.checkpoint_sha256)
        # Establish supported structure for requested members without decoding.
        self._members = {}
        with tarfile.open(fileobj=io.BytesIO(self._archive), mode="r:gz") as archive:
            for member in archive.getmembers():
                if member.name in self._members:
                    raise ValueError("Duplicate archive member.")
                if member.name in {context.authorize(i).archive_member for i in self.indices}:
                    if not member.isfile() or not 0 < member.size <= 4_000_000:
                        raise ValueError("Unsupported archive member.")
                    with archive.extractfile(member) as handle:
                        raw = handle.read()  # Serialized bytes only; no unpickling.
                    inspect_member_layout(io.BytesIO(raw), len(raw))
                    self._members[member.name] = raw
        if set(self._members) != {context.authorize(i).archive_member for i in self.indices}:
            raise ValueError("Missing authorized archive member.")
        self.profile = {"mode": "real", "origin": "authorized_validation_row",
                        "scientific_validation_evaluated": True, "notice": REAL_NOTICE,
                        "namespace": "real_validation", "execution": self.request}

    def _checked_bytes(self, reference, digest):
        path = (self.context.root / reference).resolve()
        if not path.is_relative_to(self.context.root):
            raise PermissionError("Provenance reference escapes project.")
        raw = path.read_bytes()
        if data.digest(raw) != digest:
            raise ValueError("Frozen execution provenance hash mismatch.")
        return raw

    def _revalidate(self):
        validate_approval(self.context, self.indices, self._max_images, self._approval)
        self.context.check_protected()

    def image(self, index):
        entry = self.context.authorize(index)  # TEST rejection precedes byte access.
        if index not in self.indices:
            raise PermissionError("Image is outside the explicitly approved execution subset.")
        self._revalidate()
        if (entry.archive_member != data.MEMBERS[index//88] or entry.member_row != index % 88):
            raise ValueError("Frozen index-to-row mapping mismatch.")
        raw = self._members[entry.archive_member]
        pixels = decode_authorized_row(io.BytesIO(raw), len(raw), entry.member_row,
                                       expected_label=entry.ground_truth_label,
                                       expected_filename=entry.original_filename)
        image = pixels.astype(np.float32) / np.float32(255)
        # Exact equivalence to existing trusted uint8 validation ToTensor/Normalize.
        expected = preprocess.build_transforms(preprocess.PreprocessConfig())[1](pixels).unsqueeze(0)
        if not torch.equal(faith.normalize_intensity(image), expected):
            raise ValueError("Frozen preprocessing compatibility failure.")
        return image

    def model(self):
        self._revalidate()
        if self._model is None:
            # No unsafe fallback; arbitrary serialized classes are unsupported.
            checkpoint = torch.load(io.BytesIO(self._checkpoint), map_location="cpu", weights_only=True)
            state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else None
            if not isinstance(state, dict) or not state or any(
                    not isinstance(k, str) or not isinstance(v, torch.Tensor) or not torch.isfinite(v).all()
                    for k, v in state.items()):
                raise ValueError("Unsupported/non-finite frozen model state.")
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(42)
                model = resnet_cnn.ResNetCNN()
            model.load_state_dict(state, strict=True)
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            self._model = model
        return self._model

    def diagnostic_scorer(self, model):
        """Frozen OOD diagnostics alongside classifier curves; NEVER filter them."""
        from src.ood.calibration import load_calibration
        from src.ood.ood_detector import OODDetector, OODThresholds
        self._revalidate()
        if model is not self.model():
            raise PermissionError("Only this adapter's private model is permitted.")
        path = self.context.root / "outputs/ood/ood_calibration.pt"
        relative = str(path.relative_to(self.context.root)).replace("\\", "/")
        if relative not in self.context.protected:
            raise ValueError("Frozen OOD artifact missing from protected provenance.")
        artifact = load_calibration(path, self.context.root / self.context.access.validation.provenance.checkpoint_reference)
        self._revalidate()
        detector = OODDetector(model)
        detector.feature_mean, detector.feature_variance = artifact["feature_mean"], artifact["feature_variance"]
        detector.thresholds = OODThresholds(**artifact["thresholds"])
        scorer = faith.ModelScorer(model)
        diagnostics = []

        def score(tensor):
            diagnostics.append(detector.validate_tensor(tensor))
            return scorer(tensor)

        return score, diagnostics


def memory_snapshot():
    """Best-effort process memory, separate from deterministic scientific JSON."""
    if __import__("os").name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                    "PagefileUsage", "PeakPagefileUsage")]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return {"working_set_bytes": counters.WorkingSetSize, "process_peak_bytes": counters.PeakWorkingSetSize}
    return {"status": "unavailable", "reason": "process_memory_counter_unavailable"}


def instrument(operation):
    start, cpu = time.perf_counter(), time.process_time()
    before = memory_snapshot()
    result = operation()
    return result, {"elapsed_seconds": time.perf_counter()-start, "cpu_seconds": time.process_time()-cpu,
                    "memory_before": before, "memory_after": memory_snapshot(),
                    "peak_scope": "process lifetime, not isolated image allocation"}


@dataclass(frozen=True)
class ByteRange:
    offset: int
    length: int


@dataclass
class _Dtype:
    code: str
    state: tuple | None = None


@dataclass
class _Array:
    state: tuple | None = None


@dataclass(frozen=True)
class _Scalar:
    dtype: _Dtype
    data: ByteRange


@dataclass(frozen=True)
class MemberLayout:
    rows: tuple[ByteRange, ...]
    labels: tuple[int, ...]
    filenames: tuple[object, ...]


def inspect_member_layout(stream, size):
    """Inspect metadata and seek past ALL binary arguments, including images.

    Never call pickle.loads, numpy constructors, import serialized names, or
    execute serialized instructions. Accept only the observed list of 88
    separately encoded C-order uint8 150x150 images. Reject monolithic batches,
    object arrays, overlapping row spans, unknown globals/opcodes and ambiguity.
    The supplied stream must be seekable and bounded to the archive member.
    """
    if not 0 < size <= 4_000_000:
        raise ValueError("Unsupported serialized member size.")
    stack, memo = [], {}
    mark = object()
    allowed = {("numpy.core.multiarray", "_reconstruct"), ("numpy", "ndarray"),
               ("numpy", "dtype"), ("numpy.core.multiarray", "scalar")}

    def read(n):
        if n < 0 or stream.tell()+n > size:
            raise ValueError("Truncated serialized metadata.")
        raw = stream.read(n)
        if len(raw) != n:
            raise ValueError("Truncated serialized metadata.")
        return raw

    def integer(n, signed=False):
        return int.from_bytes(read(n), "little", signed=signed)

    def marked():
        if mark not in stack:
            raise ValueError("Missing symbolic mark.")
        start = max(i for i, item in enumerate(stack) if item is mark)
        values = stack[start+1:]
        del stack[start:]
        return values

    def peek_bytes(span, bound):
        if not isinstance(span, ByteRange) or span.length > bound:
            raise ValueError("Unsupported metadata byte field.")
        position = stream.tell()
        stream.seek(span.offset)
        value = read(span.length)
        stream.seek(position)
        return value

    if read(2) != b"\x80\x03":
        raise ValueError("Only verified protocol-3 row encoding is supported.")
    try:
        for _ in range(50000):
            op = read(1)
            if op == b"}":
                stack.append({})
            elif op == b"]":
                stack.append([])
            elif op == b"(":
                stack.append(mark)
            elif op == b"N":
                stack.append(None)
            elif op in (b"\x89", b"\x88"):
                stack.append(op == b"\x88")
            elif op in (b"K", b"J", b"M"):
                stack.append(integer({b"K": 1, b"J": 4, b"M": 2}[op], signed=op == b"J"))
            elif op == b"X":
                length = integer(4)
                if length > 512:
                    raise ValueError("Oversized text metadata.")
                stack.append(read(length).decode("utf-8"))
            elif op in (b"B", b"C"):
                length = integer(4 if op == b"B" else 1)
                if length > 22500 or stream.tell()+length > size:
                    raise ValueError("Bulk/unsupported binary representation.")
                stack.append(ByteRange(stream.tell(), length))
                stream.seek(length, 1)  # NEVER read image arguments here.
            elif op == b"c":
                names = []
                for _ in range(2):
                    text = bytearray()
                    while True:
                        char = read(1)
                        if char == b"\n":
                            break
                        text.extend(char)
                        if len(text) > 64:
                            raise ValueError("Oversized global name.")
                    names.append(text.decode("ascii"))
                name = tuple(names)
                if name not in allowed:
                    raise ValueError("Unsupported serialized global (never executed).")
                stack.append(name)
            elif op in (b"q", b"r"):
                key = integer(1 if op == b"q" else 4)
                if key in memo or key > 10000 or not stack:
                    raise ValueError("Invalid symbolic memo.")
                memo[key] = stack[-1]
            elif op in (b"h", b"j"):
                stack.append(memo[integer(1 if op == b"h" else 4)])
            elif op in (b"\x85", b"\x86", b"\x87"):
                n = {b"\x85": 1, b"\x86": 2, b"\x87": 3}[op]
                if len(stack) < n:
                    raise ValueError("Invalid symbolic tuple.")
                value = tuple(stack[-n:])
                del stack[-n:]
                stack.append(value)
            elif op == b"t":
                values = marked()
                if len(values) > 8:
                    raise ValueError("Unsupported symbolic tuple.")
                stack.append(tuple(values))
            elif op == b"e":
                values = marked()
                if type(stack[-1]) is not list or len(stack[-1])+len(values) > 88:
                    raise ValueError("Unsupported symbolic list.")
                stack[-1].extend(values)
            elif op == b"u":
                values = marked()
                if type(stack[-1]) is not dict or len(values) % 2:
                    raise ValueError("Invalid member dictionary.")
                for k, v in zip(values[::2], values[1::2]):
                    if k not in {"batch_label", "labels", "data", "filenames"} or k in stack[-1]:
                        raise ValueError("Unknown/duplicate member field.")
                    stack[-1][k] = v
            elif op == b"R":
                args, constructor = stack.pop(), stack.pop()
                if not isinstance(args, tuple):
                    raise ValueError("Unsupported symbolic REDUCE arguments.")
                if constructor == ("numpy", "dtype") and len(args) == 3 and args[1:] == (False, True):
                    if args[0] != "u1" and not re.fullmatch(r"U[1-9][0-9]?", str(args[0])):
                        raise ValueError("Unsupported dtype.")
                    stack.append(_Dtype(args[0]))
                elif constructor == ("numpy.core.multiarray", "_reconstruct") and len(args) == 3:
                    if args[:2] != (("numpy", "ndarray"), (0,)) or peek_bytes(args[2], 1) != b"b":
                        raise ValueError("Unsupported ndarray description.")
                    stack.append(_Array())
                elif constructor == ("numpy.core.multiarray", "scalar") and len(args) == 2:
                    dtype, blob = args
                    if not isinstance(dtype, _Dtype) or not dtype.code.startswith("U") or not isinstance(blob, ByteRange):
                        raise ValueError("Only filename Unicode scalars are supported.")
                    if blob.length != int(dtype.code[1:])*4:
                        raise ValueError("Filename byte size mismatch.")
                    stack.append(_Scalar(dtype, blob))
                else:
                    raise ValueError("Unsupported symbolic constructor; none executed.")
            elif op == b"b":
                state = stack.pop()
                value = stack[-1]
                if not isinstance(value, (_Array, _Dtype)) or value.state is not None or type(state) is not tuple:
                    raise ValueError("Unsupported symbolic BUILD.")
                if isinstance(value, _Dtype):
                    if value.code == "u1" and state != (3, "|", None, None, None, -1, -1, 0):
                        raise ValueError("Unsupported uint8 dtype state.")
                    if value.code.startswith("U") and (len(state) != 8 or state[0] != 3 or state[1] != "<"):
                        raise ValueError("Unsupported filename dtype state.")
                value.state = state
            elif op == b".":
                if stream.tell() != size or len(stack) != 1 or type(stack[0]) is not dict:
                    raise ValueError("Trailing or incomplete member structure.")
                result = stack[0]
                if set(result) != {"batch_label", "labels", "data", "filenames"}:
                    raise ValueError("Incomplete member dictionary.")
                arrays, labels, filenames = (result[k] for k in ("data", "labels", "filenames"))
                if any(type(v) is not list or len(v) != 88 for v in (arrays, labels, filenames)):
                    raise ValueError("Require exactly 88 separately encoded rows.")
                if any(type(v) is not int or v not in (0, 1) for v in labels):
                    raise ValueError("Invalid class labels.")
                spans = []
                for array in arrays:
                    if not isinstance(array, _Array) or array.state is None or len(array.state) != 5:
                        raise ValueError("Unsupported row array structure.")
                    version, shape, dtype, fortran, blob = array.state
                    if (version != 1 or shape != (150, 150) or fortran is not False or
                            not isinstance(dtype, _Dtype) or dtype.code != "u1" or dtype.state is None or
                            not isinstance(blob, ByteRange) or blob.length != 22500):
                        raise ValueError("Only separate C-order 150x150 uint8 rows are supported.")
                    spans.append(blob)
                if any(a.offset+a.length > b.offset for a, b in zip(spans, spans[1:])):
                    raise ValueError("Aliased, overlapping or reordered image payloads.")
                if any(not isinstance(f, (str, _Scalar)) for f in filenames):
                    raise ValueError("Unsupported filename metadata.")
                return MemberLayout(tuple(spans), tuple(labels), tuple(filenames))
            else:
                raise ValueError(f"Unsupported serialized opcode {op.hex()}; no fallback loader.")
            if len(stack) > 256:
                raise ValueError("Symbolic stack limit exceeded.")
    except (IndexError, KeyError, TypeError, UnicodeError, struct.error) as error:
        raise ValueError("Malformed frozen row representation.") from error
    raise ValueError("Missing STOP or operation limit exceeded.")
