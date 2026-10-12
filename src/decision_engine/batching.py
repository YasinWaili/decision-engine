"""Candidate tokenization with injected tokenizers and optional tensor conversion.

Core batching requires no third-party packages. A compatible tokenizer accepts
a passage and text_pair, returning input_ids and attention_mask as integer lists.
"""

from dataclasses import dataclass
from typing import Any, Sequence

from .dataset import Example
from .schemas import DecisionRequest


@dataclass
class CandidateBatch:
    input_ids: list[list[list[int]]]
    attention_mask: list[list[list[int]]]
    option_mask: list[list[bool]]
    example_ids: list[str]
    targets: list[int] | None = None
    token_type_ids: list[list[list[int]]] | None = None

    @property
    def shape(self) -> tuple[int, int, int]:
        return (len(self.input_ids), len(self.input_ids[0]), len(self.input_ids[0][0]))

    def to_torch(self, device: str = "cpu") -> dict[str, Any]:
        """Create model inputs lazily; no automatic GPU selection or training."""
        import torch

        result = {
            "input_ids": torch.tensor(self.input_ids, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(self.attention_mask, dtype=torch.long, device=device),
            "option_mask": torch.tensor(self.option_mask, dtype=torch.bool, device=device),
        }
        if self.token_type_ids is not None:
            result["token_type_ids"] = torch.tensor(self.token_type_ids, dtype=torch.long, device=device)
        return result

    def targets_to_torch(self, device: str = "cpu") -> Any:
        import torch

        if self.targets is None:
            raise ValueError("Inference batches have no training targets")
        return torch.tensor(self.targets, dtype=torch.long, device=device)


class CandidateCollator:
    """Dynamic right-padding for tokens and variable option counts.

    Pair layout: [passage] [separator] Question: ... Candidate answer: ...
    Only the passage may be truncated. Too-long question/answer pairs fail
    instead of silently dropping the candidate or claim. Token limits are
    checked independently of any tokenizer's own truncation implementation.
    """

    def __init__(self, tokenizer: Any, max_length: int = 128) -> None:
        if type(max_length) is not int or max_length < 16:
            raise ValueError("max_length must be an integer of at least 16")
        if type(getattr(tokenizer, "pad_token_id", None)) is not int:
            raise ValueError("Tokenizer needs an integer pad_token_id")
        model_limit = getattr(tokenizer, "model_max_length", None)
        if isinstance(model_limit, int) and model_limit < max_length:
            raise ValueError("max_length exceeds the tokenizer's model_max_length")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def examples(self, examples: Sequence[Example]) -> CandidateBatch:
        if not examples:
            raise ValueError("Cannot batch an empty example list")
        batch = self.requests([example.request() for example in examples])
        batch.example_ids = [example.id for example in examples]
        batch.targets = [example.target_index for example in examples]
        return batch

    def requests(self, requests: Sequence[DecisionRequest]) -> CandidateBatch:
        if not requests:
            raise ValueError("Cannot batch an empty request list")
        encoded = []
        for request_index, request in enumerate(requests):
            candidates = []
            for option in request.options:
                pair = f"Question: {request.question}\nCandidate answer: {option}"
                try:
                    result = self.tokenizer(
                        request.state, text_pair=pair, padding=False,
                        truncation="only_first", max_length=self.max_length,
                        return_attention_mask=True,
                    )
                except Exception as error:
                    # Rust-backed fast tokenizers raise plain Exception for an
                    # impossible only_first truncation. Normalize errors only
                    # at this external tokenizer-call boundary for CLI callers.
                    raise ValueError(f"Request {request_index}, option {option!r}: {error}") from error
                ids = list(result["input_ids"])
                mask = list(result["attention_mask"])
                if not ids or len(ids) != len(mask) or len(ids) > self.max_length:
                    raise ValueError("Tokenizer returned invalid lengths; shorten the question or increase max_length")
                if any(type(token) is not int or token < 0 for token in ids):
                    raise ValueError("Tokenizer input_ids must be nonnegative integers")
                if any(type(value) is not int or value != 1 for value in mask):
                    raise ValueError("Tokenizer must return unpadded candidates with attention_mask=1")
                types = result.get("token_type_ids")
                if types is not None:
                    types = list(types)
                    if len(types) != len(ids) or any(type(t) is not int or t < 0 for t in types):
                        raise ValueError("Tokenizer returned invalid token_type_ids")
                candidates.append((ids, mask, types))
            encoded.append(candidates)
        use_types = encoded[0][0][2] is not None
        if any((candidate[2] is not None) != use_types for row in encoded for candidate in row):
            raise ValueError("Tokenizer returned inconsistent token_type_ids")
        options = max(len(row) for row in encoded)
        tokens = max(len(candidate[0]) for row in encoded for candidate in row)
        all_ids, all_masks, all_types, option_masks = [], [], [], []
        for row in encoded:
            ids_row, mask_row, types_row = [], [], []
            for ids, mask, types in row:
                padding = tokens - len(ids)
                ids_row.append(ids + [self.tokenizer.pad_token_id] * padding)
                mask_row.append(mask + [0] * padding)
                if use_types:
                    types_row.append(types + [0] * padding)
            missing = options - len(row)
            ids_row.extend([[self.tokenizer.pad_token_id] * tokens for _ in range(missing)])
            mask_row.extend([[0] * tokens for _ in range(missing)])
            if use_types:
                types_row.extend([[0] * tokens for _ in range(missing)])
            all_ids.append(ids_row)
            all_masks.append(mask_row)
            all_types.append(types_row)
            option_masks.append([True] * len(row) + [False] * missing)
        return CandidateBatch(all_ids, all_masks, option_masks,
                              [f"request-{i}" for i in range(len(requests))],
                              token_type_ids=all_types if use_types else None)


def load_tokenizer(name_or_path: str, allow_download: bool = False) -> Any:
    """Load a Transformers tokenizer; offline/local-cache-only by default."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        name_or_path, local_files_only=not allow_download, trust_remote_code=False,
    )
