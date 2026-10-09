"""Predictor interface and explicit mock; no model downloads."""

from typing import Protocol

from .schemas import DecisionRequest, DecisionResponse


class Predictor(Protocol):
    def predict(self, request: DecisionRequest) -> DecisionResponse: ...


class MockPredictor:
    """Uniform fake probabilities for application testing, not intelligence."""

    def predict(self, request: DecisionRequest) -> DecisionResponse:
        return DecisionResponse(
            decision=request.options[0],
            probabilities={option: 1 / len(request.options) for option in request.options},
            abstained=True,
            source="mock_uniform_not_a_trained_model",
        )
