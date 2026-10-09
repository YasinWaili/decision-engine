"""Dependency-free probability metrics and tie-aware selective prediction."""

import math

from .dataset import Example
from .predictors import Predictor
from .schemas import DecisionResponse


def evaluate(
    examples: list[Example], responses: list[DecisionResponse], bins: int = 10,
) -> dict:
    if not examples or len(examples) != len(responses):
        raise ValueError("Need one response per example in a nonempty dataset")
    if type(bins) is not int or bins < 1:
        raise ValueError("bins must be a positive integer")
    records = []
    brier = nll = 0.0
    for example, response in zip(examples, responses):
        if set(response.probabilities) != set(example.options):
            raise ValueError(f"{example.id}: response options do not match the example")
        # Recheck distributions in case the dict was mutated after construction.
        DecisionResponse(**response.to_dict())
        probabilities = [response.probabilities[option] for option in example.options]
        correct = int(response.decision == example.options[example.target_index])
        confidence = max(probabilities)
        brier += sum((p - int(i == example.target_index)) ** 2 for i, p in enumerate(probabilities))
        # Zero target probability is clipped solely for finite reporting.
        nll -= math.log(max(probabilities[example.target_index], 1e-15))
        records.append((confidence, correct, response.abstained))
    n = len(records)
    reliability = []
    ece = 0.0
    for index in range(bins):
        rows = [r for r in records if min(int(r[0] * bins), bins - 1) == index]
        count = len(rows)
        mean_confidence = sum(r[0] for r in rows) / count if count else None
        accuracy = sum(r[1] for r in rows) / count if count else None
        if count:
            ece += count / n * abs(mean_confidence - accuracy)
        reliability.append({"lower": index / bins, "upper": (index + 1) / bins,
                            "count": count, "mean_probability": mean_confidence, "accuracy": accuracy})
    # Include entire confidence ties: never imply arbitrary within-tie ordering
    # is achievable by a probability threshold.
    curve = [{"threshold": None, "coverage": 0.0, "error_rate": None}]
    selected = errors = 0
    by_confidence: dict[float, list[int]] = {}
    for confidence, correct, _ in records:
        by_confidence.setdefault(confidence, []).append(correct)
    area = 0.0
    for confidence in sorted(by_confidence, reverse=True):
        outcomes = by_confidence[confidence]
        selected += len(outcomes)
        errors += len(outcomes) - sum(outcomes)
        risk = errors / selected
        area += len(outcomes) / n * risk
        curve.append({"threshold": confidence, "coverage": selected / n, "error_rate": risk})
    acted = [r for r in records if not r[2]]
    return {
        "examples": n,
        "accuracy": sum(r[1] for r in records) / n,
        "brier_score": brier / n,
        "negative_log_likelihood": nll / n,
        "expected_calibration_error": ece,
        "reliability_bins": reliability,
        "risk_coverage_curve": curve,
        "risk_coverage_area_step": area,
        "policy": {"coverage": len(acted) / n,
                   "error_rate": sum(1 - r[1] for r in acted) / len(acted) if acted else None},
        "sources": sorted({r.source for r in responses}),
        "all_outputs_marked_calibrated": all(r.calibrated for r in responses),
        "notes": [
            "Brier score uses the summed multiclass convention (range 0 to 2).",
            "NLL clips target probabilities at 1e-15; natural logarithms.",
            "ECE is top-label, equal-width binned and depends on bin choice/sample size.",
            "Risk coverage uses top probability with whole ties; area uses right-step integration.",
            "Policy coverage respects abstention; risk coverage explores potential thresholds separately.",
            "A calibrated flag is metadata, not proof of calibration.",
            "Never select deployment thresholds on the final test set.",
        ],
    }


def evaluate_predictor(examples: list[Example], predictor: Predictor, bins: int = 10) -> dict:
    return evaluate(examples, [predictor.predict(e.request()) for e in examples], bins)
