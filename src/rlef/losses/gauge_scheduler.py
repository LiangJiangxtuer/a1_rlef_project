from __future__ import annotations


def warm_gauge_anchor_weight(
    step: int,
    ramp_start: int = 300,
    full_start: int = 700,
    max_weight: float = 0.005,
    hard_cap: float = 0.010,
) -> float:
    """Warm-start schedule for gauge anchoring.

    The default schedule keeps gauge anchoring off during early shape learning,
    ramps conservatively, and hard-caps the anchor below rejected P7c strengths.
    """
    max_weight = float(max_weight)
    hard_cap = float(hard_cap)
    if max_weight > hard_cap:
        raise ValueError(f'max_weight={max_weight} exceeds hard cap {hard_cap}')
    if full_start <= ramp_start:
        raise ValueError('full_start must be greater than ramp_start')
    step = int(step)
    if step <= ramp_start:
        return 0.0
    if step >= full_start:
        return max_weight
    progress = (step - ramp_start) / float(full_start - ramp_start)
    return max_weight * progress
