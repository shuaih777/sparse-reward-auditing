"""Sentinel-repair experiments for verifier-noise in group-relative RL."""

from .advantages import (
    DEFAULT_DDOF,
    DEFAULT_EPSILON,
    group_advantages,
    grpo_advantages,
    validate_binary_rewards,
)
from .harm import (
    RepairMode,
    RepairOutcome,
    ResidualNorm,
    apply_repair,
    coefficient_residual,
    marginal_repair_effect,
    residual_after_repair,
    reward_coefficient_residual,
)

__all__ = (
    "DEFAULT_DDOF",
    "DEFAULT_EPSILON",
    "RepairMode",
    "RepairOutcome",
    "ResidualNorm",
    "apply_repair",
    "coefficient_residual",
    "grpo_advantages",
    "group_advantages",
    "marginal_repair_effect",
    "residual_after_repair",
    "reward_coefficient_residual",
    "validate_binary_rewards",
)
