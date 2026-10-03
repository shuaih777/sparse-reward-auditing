"""Public data views and loaders for logged GRPO rollouts.

The important boundary in this module is that :class:`LoggedRollout` never
contains the oracle label.  ``ParsedLog`` owns the labels privately and can
mint an oracle and an evaluator; selectors are only ever handed observation
objects built from the public rollouts.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Iterator, Mapping, Sequence


def _as_finite_float(value: Any, *, column: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{column} must be numeric, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{column} must be finite, got {value!r}")
    return parsed


def _optional_float(value: Any, *, column: str) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return _as_finite_float(value, column=column)


@dataclass(frozen=True)
class LoggedRollout:
    """One rollout containing only information available before an audit."""

    rollout_id: int
    step: str
    prompt: str
    completion: str
    cheap_reward: float
    logged_advantage: float | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rollout_id < 0:
            raise ValueError("rollout_id must be non-negative")
        if not math.isfinite(float(self.cheap_reward)):
            raise ValueError("cheap_reward must be finite")
        if self.logged_advantage is not None and not math.isfinite(
            float(self.logged_advantage)
        ):
            raise ValueError("logged_advantage must be finite when present")
        # Prevent a caller from mutating a metadata dictionary after parsing.
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class RolloutGroup:
    """A *consecutive* run of rows with the same (step, prompt)."""

    group_id: int
    step: str
    prompt: str
    rollouts: tuple[LoggedRollout, ...]

    def __post_init__(self) -> None:
        if not self.rollouts:
            raise ValueError("a rollout group cannot be empty")
        if any(
            row.step != self.step or row.prompt != self.prompt for row in self.rollouts
        ):
            raise ValueError("all rows in a group must share its step and prompt")
        ids = [row.rollout_id for row in self.rollouts]
        if len(ids) != len(set(ids)):
            raise ValueError("rollout ids must be unique within a group")

    def __len__(self) -> int:
        return len(self.rollouts)

    @property
    def cheap_rewards(self) -> tuple[float, ...]:
        return tuple(row.cheap_reward for row in self.rollouts)


@dataclass(frozen=True)
class RolloutObservation:
    """What a selector may know about one rollout at a particular instant.

    ``current_reward`` equals the cheap reward until this rollout is audited;
    afterwards it equals the purchased label.  There deliberately is no
    ``oracle_reward`` attribute.
    """

    rollout_id: int
    group_id: int
    position: int
    step: str
    prompt: str
    completion: str
    cheap_reward: float
    current_reward: float
    advantage: float
    audited: bool
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class GroupObservation:
    """A recomputed, oracle-safe view of a rollout group."""

    group_id: int
    step: str
    prompt: str
    rollouts: tuple[RolloutObservation, ...]
    version: int = 0
    quarantined: bool = False

    def __post_init__(self) -> None:
        if not self.rollouts:
            raise ValueError("an observed group cannot be empty")
        if any(row.group_id != self.group_id for row in self.rollouts):
            raise ValueError("observed rollout has the wrong group_id")

    def __len__(self) -> int:
        return len(self.rollouts)

    @property
    def unqueried(self) -> tuple[RolloutObservation, ...]:
        return tuple(row for row in self.rollouts if not row.audited)

    @property
    def current_rewards(self) -> tuple[float, ...]:
        return tuple(row.current_reward for row in self.rollouts)

    @property
    def advantages(self) -> tuple[float, ...]:
        return tuple(row.advantage for row in self.rollouts)


def group_consecutive_rollouts(
    rows: Iterable[LoggedRollout],
) -> tuple[RolloutGroup, ...]:
    """Create groups without merging non-contiguous occurrences of a prompt.

    A global ``groupby(prompt)`` is wrong for these logs: the same prompt can
    occur again later (including at a later policy step) and must then form a
    new policy group.  This function starts a new group whenever the current
    ``(step, prompt)`` differs from the immediately preceding row.
    """

    groups: list[RolloutGroup] = []
    current: list[LoggedRollout] = []
    previous_key: tuple[str, str] | None = None

    def flush() -> None:
        if not current:
            return
        first = current[0]
        groups.append(
            RolloutGroup(
                group_id=len(groups),
                step=first.step,
                prompt=first.prompt,
                rollouts=tuple(current),
            )
        )

    for row in rows:
        key = (row.step, row.prompt)
        if previous_key is not None and key != previous_key:
            flush()
            current = []
        current.append(row)
        previous_key = key
    flush()
    return tuple(groups)


_DEFAULT_PRIVATE_COLUMNS = frozenset(
    {
        "oracle_reward",
        "oracle_label",
        "ground_truth",
        "ground_truth_reward",
        "gt_answer",
        "gold_answer",
        "target",
    }
)


@dataclass(frozen=True)
class ParsedLog:
    """A parsed log with a public trajectory and privately held truth."""

    groups: tuple[RolloutGroup, ...]
    _truth_items: tuple[tuple[int, float], ...] = field(repr=False)

    @property
    def rollout_count(self) -> int:
        return sum(len(group) for group in self.groups)

    def make_oracle(self):
        """Return a fresh query-only oracle for this log."""

        from .oracle import HiddenOracle

        return HiddenOracle(dict(self._truth_items))

    def make_evaluator(self):
        """Return a fresh aggregate evaluator for this log."""

        from .oracle import HiddenEvaluator

        return HiddenEvaluator(dict(self._truth_items))


def parse_logged_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    reward_column: str = "reward",
    oracle_column: str = "oracle_reward",
    advantage_column: str = "advantage",
    prompt_column: str = "prompt",
    completion_column: str = "completion",
    step_column: str = "step",
    private_columns: Iterable[str] = (),
    metadata_columns: Iterable[str] = (),
) -> ParsedLog:
    """Parse mappings into public groups plus privately held oracle labels.

    Metadata is fail-closed: no extra input column is visible to a selector
    unless its name is explicitly listed in ``metadata_columns``.  A denylist
    is unsafe here because names such as ``is_correct`` or ``clean_reward`` can
    disclose truth without containing the word ``oracle``.
    """

    private = set(_DEFAULT_PRIVATE_COLUMNS)
    private.update(private_columns)
    private.add(oracle_column)
    allowed_metadata = set(metadata_columns)
    private_lower = {name.lower() for name in private}

    def is_private_name(name: str) -> bool:
        lowered = name.lower()
        return (
            lowered in private_lower
            or lowered.startswith(
                ("oracle_", "gold_", "ground_truth", "gt_", "trigger_")
            )
            or "is_correct" in lowered
            or "clean_reward" in lowered
        )

    forbidden_requested = {name for name in allowed_metadata if is_private_name(name)}
    if forbidden_requested:
        raise ValueError(
            f"metadata_columns requests private fields: {sorted(forbidden_requested)}"
        )
    public_core = {
        reward_column,
        advantage_column,
        prompt_column,
        completion_column,
        step_column,
    }

    public_rows: list[LoggedRollout] = []
    truth: list[tuple[int, float]] = []
    for rollout_id, raw in enumerate(rows):
        if reward_column not in raw:
            raise ValueError(f"missing required column {reward_column!r}")
        if oracle_column not in raw:
            raise ValueError(f"missing required column {oracle_column!r}")
        if prompt_column not in raw:
            raise ValueError(f"missing required column {prompt_column!r}")

        reward = _as_finite_float(raw[reward_column], column=reward_column)
        oracle_reward = _as_finite_float(raw[oracle_column], column=oracle_column)
        advantage = _optional_float(raw.get(advantage_column), column=advantage_column)
        step = str(raw.get(step_column, ""))
        prompt = str(raw[prompt_column])
        completion = str(raw.get(completion_column, ""))

        metadata: dict[str, str] = {}
        for key, value in raw.items():
            lower_key = key.lower()
            if key in public_core or key in private or key not in allowed_metadata:
                continue
            # Also block custom columns whose names unmistakably disclose
            # oracle/gold/ground-truth information.
            if is_private_name(lower_key):
                continue
            metadata[key] = "" if value is None else str(value)

        public_rows.append(
            LoggedRollout(
                rollout_id=rollout_id,
                step=step,
                prompt=prompt,
                completion=completion,
                cheap_reward=reward,
                logged_advantage=advantage,
                metadata=metadata,
            )
        )
        truth.append((rollout_id, oracle_reward))

    return ParsedLog(
        groups=group_consecutive_rollouts(public_rows),
        _truth_items=tuple(truth),
    )


def _natural_key(path: Path) -> tuple[Any, ...]:
    parts = re.split(r"(\d+)", path.name)
    return tuple(int(part) if part.isdigit() else part for part in parts)


def load_debug_logs(path: str | Path, **parse_kwargs: Any) -> ParsedLog:
    """Load one TSV or a directory of TSVs in natural filename order."""

    source = Path(path)
    if source.is_dir():
        files = sorted(source.glob("*.tsv"), key=_natural_key)
    else:
        files = [source]
    if not files:
        raise FileNotFoundError(f"no TSV logs found under {source}")

    records: list[dict[str, str]] = []
    step_column = str(parse_kwargs.get("step_column", "step"))
    for file_index, file_path in enumerate(files):
        with file_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames is None:
                raise ValueError(f"TSV has no header: {file_path}")
            for row in reader:
                # Some exported per-step files omit the step column.  A file
                # boundary is a real policy-time boundary, so preserve it.
                if not row.get(step_column):
                    row[step_column] = str(file_index)
                records.append(row)
    return parse_logged_rows(records, **parse_kwargs)


def iter_contiguous_step_batches(
    groups: Sequence[RolloutGroup],
) -> Iterator[tuple[RolloutGroup, ...]]:
    """Yield consecutive policy-step batches without looking into the future.

    If a group has no step identifier, it is conservatively treated as its own
    arrival batch rather than making the full dataset simultaneously visible.
    """

    current: list[RolloutGroup] = []
    current_step: str | None = None
    for group in groups:
        if group.step == "":
            if current:
                yield tuple(current)
                current = []
                current_step = None
            yield (group,)
            continue
        if current and group.step != current_step:
            yield tuple(current)
            current = []
        current.append(group)
        current_step = group.step
    if current:
        yield tuple(current)
