"""Dependency-free input and output contracts."""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class DecisionRequest:
    state: str
    question: str
    options: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("state", "question"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        if not isinstance(self.options, tuple):
            raise ValueError("options must be a tuple")
        if not 2 <= len(self.options) <= 10:
            raise ValueError("Provide between 2 and 10 options")
        if any(not isinstance(option, str) or not option.strip() for option in self.options):
            raise ValueError("Every option must be a nonempty string")
        normalized = [option.strip().casefold() for option in self.options]
        if len(set(normalized)) != len(normalized):
            raise ValueError("Options must be distinct, ignoring case and outer whitespace")


@dataclass(frozen=True)
class DecisionResponse:
    decision: str
    probabilities: dict[str, float]
    abstained: bool
    source: str
    calibrated: bool = False

    def __post_init__(self) -> None:
        if len(self.probabilities) < 2 or self.decision not in self.probabilities:
            raise ValueError("Decision must belong to a distribution with at least two options")
        values = list(self.probabilities.values())
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in values):
            raise ValueError("Probabilities must be finite and between zero and one")
        if not math.isclose(sum(values), 1.0, abs_tol=1e-6):
            raise ValueError("Probabilities must sum to one")
        if self.probabilities[self.decision] != max(values):
            raise ValueError("Decision must have the highest probability")

    def to_dict(self) -> dict:
        return asdict(self)
