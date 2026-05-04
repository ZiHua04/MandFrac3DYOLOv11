from __future__ import annotations

import json
from typing import Any, Sequence


def coerce_qa_alpha_per_level(value: Any) -> tuple[float, ...] | None:
    """Coerce config/CLI values into a normalized alpha tuple or None."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            value = json.loads(text)
        except Exception as exc:  # pragma: no cover - defensive parsing
            raise ValueError(f"Invalid qa_alpha_per_level string: {value}") from exc
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        raise ValueError(f"qa_alpha_per_level must be a sequence, got {type(value).__name__}")
    values = tuple(float(v) for v in value)
    if not values:
        return None
    for alpha in values:
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError(f"Each qa_alpha_per_level value must be in [0, 1], got {alpha}")
    return values


def resolve_qa_alpha_per_level(
    qa_alpha: float,
    qa_alpha_per_level: Sequence[float] | None,
    num_levels: int,
) -> tuple[float, ...]:
    """Resolve scalar/per-level QA fusion into one alpha per detection level."""
    if num_levels <= 0:
        raise ValueError(f"num_levels must be positive, got {num_levels}")
    per_level = coerce_qa_alpha_per_level(qa_alpha_per_level)
    if per_level is not None:
        if len(per_level) != int(num_levels):
            raise ValueError(
                f"qa_alpha_per_level expects {int(num_levels)} values, got {len(per_level)}: {per_level}"
            )
        return per_level

    alpha = float(qa_alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"qa_alpha must be in [0, 1], got {alpha}")
    return tuple(alpha for _ in range(int(num_levels)))
