"""Query-only oracle and hidden aggregate evaluator.

Neither object is passed to a selector.  The oracle discloses one purchased
label at a time; the evaluator returns aggregate harm numbers only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .data import RolloutGroup


class OracleQueryError(RuntimeError):
    """Raised for unknown, duplicate, or otherwise invalid oracle calls."""


@dataclass(frozen=True)
class OracleAnswer:
    rollout_id: int
    label: float


class HiddenOracle:
    """A label store that only exposes explicitly purchased labels."""

    __slots__ = ("__truth", "__queried", "__calls")

    def __init__(self, truth: Mapping[int, float]) -> None:
        parsed = {int(key): float(value) for key, value in truth.items()}
        if any(not math.isfinite(value) for value in parsed.values()):
            raise ValueError("all oracle labels must be finite")
        self.__truth = parsed
        self.__queried: set[int] = set()
        self.__calls = 0

    @property
    def calls(self) -> int:
        return self.__calls

    @property
    def queried_ids(self) -> frozenset[int]:
        """Ids are public after purchase; labels remain query-return-only."""

        return frozenset(self.__queried)

    def query(self, rollout_id: int) -> OracleAnswer:
        return self.query_many((rollout_id,))[0]

    def query_many(self, rollout_ids: Sequence[int]) -> tuple[OracleAnswer, ...]:
        """Atomically validate a bundle, then reveal precisely that bundle."""

        ids = tuple(int(item) for item in rollout_ids)
        if not ids:
            raise OracleQueryError("an oracle query cannot be empty")
        if len(ids) != len(set(ids)):
            raise OracleQueryError("a query bundle cannot contain duplicate ids")
        unknown = [item for item in ids if item not in self.__truth]
        if unknown:
            raise OracleQueryError(f"unknown rollout ids: {unknown}")
        repeated = [item for item in ids if item in self.__queried]
        if repeated:
            raise OracleQueryError(f"rollout ids were already queried: {repeated}")

        answers = tuple(OracleAnswer(item, self.__truth[item]) for item in ids)
        self.__queried.update(ids)
        self.__calls += len(ids)
        return answers


@dataclass(frozen=True)
class AggregateEvaluation:
    group_count: int
    rollout_count: int
    coefficient_residual: float


class HiddenEvaluator:
    """Compute aggregate counterfactual metrics without disclosing labels."""

    __slots__ = ("__truth",)

    def __init__(self, truth: Mapping[int, float]) -> None:
        parsed = {int(key): float(value) for key, value in truth.items()}
        if any(not math.isfinite(value) for value in parsed.values()):
            raise ValueError("all evaluator labels must be finite")
        self.__truth = parsed

    def _oracle_rewards(self, group: RolloutGroup) -> tuple[float, ...]:
        try:
            return tuple(self.__truth[row.rollout_id] for row in group.rollouts)
        except KeyError as exc:
            raise ValueError(
                f"evaluator has no label for rollout {exc.args[0]}"
            ) from exc

    def group_residual(
        self,
        group: RolloutGroup,
        current_rewards: Sequence[float],
        *,
        norm: str = "l1",
        weights: Sequence[float] | None = None,
    ) -> float:
        """Return coefficient residual against the hidden fully-audited group."""

        if len(current_rewards) != len(group):
            raise ValueError("current_rewards must have one item per group rollout")
        from .harm import reward_coefficient_residual

        return float(
            reward_coefficient_residual(
                current_rewards,
                self._oracle_rewards(group),
                norm=norm,
                weights=weights,
            )
        )

    def group_coefficient_residual(
        self,
        group: RolloutGroup,
        coefficients: Sequence[float],
        *,
        norm: str = "l1",
        weights: Sequence[float] | None = None,
    ) -> float:
        """Score supplied update coefficients against hidden oracle coefficients."""

        if len(coefficients) != len(group):
            raise ValueError("coefficients must have one item per group rollout")
        from .advantages import grpo_advantages
        from .harm import coefficient_residual

        oracle_coefficients = grpo_advantages(self._oracle_rewards(group))
        return float(
            coefficient_residual(
                coefficients,
                oracle_coefficients,
                norm=norm,
                weights=weights,
            )
        )

    def evaluate(
        self,
        group_rewards: Iterable[tuple[RolloutGroup, Sequence[float]]],
        *,
        norm: str = "l1",
    ) -> AggregateEvaluation:
        group_count = 0
        rollout_count = 0
        residual = 0.0
        for group, rewards in group_rewards:
            group_count += 1
            rollout_count += len(group)
            residual += self.group_residual(group, rewards, norm=norm)
        return AggregateEvaluation(group_count, rollout_count, residual)

    def evaluate_cheap(
        self,
        groups: Iterable[RolloutGroup],
        *,
        norm: str = "l1",
    ) -> AggregateEvaluation:
        return self.evaluate(
            ((group, group.cheap_rewards) for group in groups),
            norm=norm,
        )

    def evaluate_coefficients(
        self,
        group_coefficients: Iterable[tuple[RolloutGroup, Sequence[float]]],
        *,
        norm: str = "l1",
    ) -> AggregateEvaluation:
        group_count = 0
        rollout_count = 0
        residual = 0.0
        for group, coefficients in group_coefficients:
            group_count += 1
            rollout_count += len(group)
            residual += self.group_coefficient_residual(group, coefficients, norm=norm)
        return AggregateEvaluation(group_count, rollout_count, residual)
