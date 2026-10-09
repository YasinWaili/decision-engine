"""Optional PyTorch model skeleton. Import requires the model dependencies.

An encoder receives each (state, question, option) candidate independently.
Its shared scoring head produces logits [batch, number_of_options].
This file does not load weights or start training automatically.
"""

import torch
from torch import nn


class CandidateScorer(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_size: int) -> None:
        super().__init__()
        self.encoder = encoder
        # Randomly initialized: meaningless predictions until trained.
        self.scoring_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        option_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Score tokenized candidates.

        input_ids / attention_mask: [batch, options, tokens]
        option_mask: [batch, options], True for real options, False for padding.
        Returns logits; padded options receive negative infinity.
        """
        if input_ids.ndim != 3 or attention_mask.shape != input_ids.shape:
            raise ValueError("Expected matching [batch, options, tokens] tensors")
        batch, options, tokens = input_ids.shape
        if option_mask.shape != (batch, options) or option_mask.dtype != torch.bool:
            raise ValueError("option_mask must be a boolean [batch, options] tensor")
        if not torch.all(option_mask.sum(dim=-1) >= 2):
            raise ValueError("Each example needs at least two valid options")
        if torch.any(attention_mask.sum(dim=-1)[option_mask] == 0):
            raise ValueError("Real candidates must contain at least one token")

        # Exclude padded candidates from the encoder entirely.
        valid = option_mask.reshape(-1)
        ids = input_ids.reshape(batch * options, tokens)[valid]
        mask = attention_mask.reshape(batch * options, tokens)[valid]
        hidden = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        scores = self.scoring_head(pooled).squeeze(-1)
        logits = scores.new_full((batch * options,), float("-inf"))
        logits[valid] = scores
        return logits.reshape(batch, options)


def probabilities_from_logits(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Softmax is not evidence of calibration; temperature must be validated."""
    import math

    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    return torch.softmax(logits / temperature, dim=-1)
