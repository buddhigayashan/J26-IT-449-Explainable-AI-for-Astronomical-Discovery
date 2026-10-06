"""AUTO candidate agreement and separately calibrated compatibility decisions."""

from typing import Any

import torch

from src.ood.auto_radio import extract_auto_candidates
from src.ood.external_compatibility import CONSISTENCY_POLICY, load_external_detector


def _rejected(reason, *, status="UNSUPPORTED", details=None, handling="auto_web"):
    return {"predicted_class": status, "confidence": None, "probabilities": {},
            "message": reason, "ood": details, "input_mode": "auto_web",
            "input_handling": handling, "representations": []}


def _probabilities(model, tensor):
    with torch.no_grad():
        logits = model(tensor)
        if logits.shape != (1, 2) or not torch.isfinite(logits).all():
            raise ValueError("Non-finite or invalid classifier output.")
        return torch.softmax(logits, dim=1)[0]


def predict_auto_image(image: Any):
    # Runtime import keeps original predict_image API independent of AUTO details.
    from src.inference.predict import CHECKPOINT_PATH, CLASS_NAMES, load_calibrated_ood_detector

    try:
        prepared = extract_auto_candidates(image)
    except (ValueError, OSError) as error:
        return _rejected(str(error))
    try:
        original_detector = load_calibrated_ood_detector()
        original_result = None
        if prepared.already_compatible:
            candidate = prepared.candidates[0]
            original_result = original_detector.validate_tensor(candidate.guarded.tensor)
            if original_result["accepted"]:
                probabilities = _probabilities(original_detector.model, candidate.guarded.tensor)
                values = {name: float(probabilities[i]) for i, name in enumerate(CLASS_NAMES)}
                label = max(values, key=values.get)
                return {"predicted_class": label, "confidence": values[label], "probabilities": values,
                        "message": "Original pixels preserved; frozen MiraBest OOD checks passed. External/web predictions remain experimental.",
                        "ood": original_result, "input_mode": "auto_web",
                        "input_handling": "preserved pixels / original MiraBest OOD",
                        "representations": [candidate.representation]}
        # A distinct validated artifact is mandatory for everything not accepted
        # by the preserved-pixel original path. Never use heuristic-only fallback.
        detector = load_external_detector(original_detector.model, CHECKPOINT_PATH)
        records, probabilities = [], []
        for candidate in prepared.candidates:
            outcome = detector.validate_tensor(candidate.guarded.tensor)
            probs = _probabilities(detector.model, candidate.guarded.tensor)
            index = int(probs.argmax())
            records.append({"representation": candidate.representation, "gate": outcome,
                            "class": CLASS_NAMES[index], "confidence": float(probs[index])})
            probabilities.append(probs)
        details = {"gate": "external_synthetic_compatibility", "original_ood": original_result,
                   "candidates": records}
        # Evaluate agreement across ALL domain-plausible representations, not
        # only the confident subset or whichever variant passes its feature gate.
        if len({r["class"] for r in records}) != 1:
            result = _rejected("Candidate radio representations produced inconsistent FR predictions.",
                               status="AMBIGUOUS", details=details)
        elif not all(r["gate"]["accepted"] for r in records):
            result = _rejected("Image does not contain a sufficiently compatible radio-morphology representation (external feature gate rejected a candidate).", details=details)
        elif min(r["confidence"] for r in records) < CONSISTENCY_POLICY["min_external_confidence"]:
            result = _rejected("Compatible representations do not provide sufficiently confident FR evidence.", details=details)
        else:
            mean = torch.stack(probabilities).mean(dim=0)
            values = {name: float(mean[i]) for i, name in enumerate(CLASS_NAMES)}
            label = records[0]["class"]
            return {"predicted_class": label, "confidence": values[label], "probabilities": values,
                    "message": "Compatible radio-map candidates agree. External/web-image predictions are experimental; radio provenance is unverified.",
                    "ood": details, "input_mode": "auto_web",
                    "input_handling": "automatic candidates / separate external compatibility gate",
                    "representations": [r["representation"] for r in records]}
        result["representations"] = [r["representation"] for r in records]
        return result
    except Exception as error:
        return _rejected("Compatibility validation is unavailable; the image cannot be classified safely.",
                         details={"error": str(error)})
