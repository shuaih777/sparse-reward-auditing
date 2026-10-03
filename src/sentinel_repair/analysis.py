"""Mechanism diagnostics for verifier-noise GRPO traces.

This module is intentionally split from the selector interface.  It may inspect
the fully logged oracle labels to evaluate a policy, while selectors only see
the masked objects constructed by :mod:`sentinel_repair.data`.
"""

from __future__ import annotations

import heapq
import itertools
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .advantages import grpo_advantages
from .controller import (
    DynamicSentinelRepairController,
    FixedSentinelRepairController,
    fixed_quota_hit_probability,
)
from .data import LoggedRollout, ParsedLog, RolloutGroup
from .group_selectors import ExpectedMarginalGroupSelector
from .harm import marginal_repair_effect, reward_coefficient_residual
from .offline import OfflineRunResult, run_parsed_log
from .selectors import (
    AbsAdvantageSelector,
    HashedLogisticRiskModel,
    RandomSelector,
    RiskWeightedAdvantageSelector,
    WholeGroupRiskSelector,
)

_STEP_FILE_RE = re.compile(r"step-(\d+)\.tsv$")

INTEGRITY_FIELDS = (
    "is_flip_target",
    "row_in_log",
    "group_in_log",
    "group_id",
    "group_position",
    "group_size",
    "label_error",
    "false_positive",
    "false_negative",
    "triggered_false_positive",
)
_STRUCTURAL_INTEGRITY_FIELDS = (
    "row_in_log",
    "group_in_log",
    "group_id",
    "group_position",
    "group_size",
)
_BOOLEAN_INTEGRITY_FIELDS = (
    "is_flip_target",
    "label_error",
    "false_positive",
    "false_negative",
    "triggered_false_positive",
)


@dataclass(frozen=True)
class LoadedTrace:
    frame: pd.DataFrame
    files: tuple[Path, ...]
    excluded_files: tuple[Path, ...]
    warnings: tuple[str, ...]
    integrity_fields: tuple[str, ...]
    used_explicit_group_ids: bool


@dataclass(frozen=True)
class TakeoverWindow:
    tau10: int | None
    tau50: int | None
    tau80: int | None
    analysis_last_step_index: int
    through_takeover_last_step_index: int
    minimum_oracle_negatives: int
    consecutive_steps: int


def _natural_path_key(path: Path) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(path))
    )


def resolve_trace_files(path: str | Path) -> tuple[Path, ...]:
    """Resolve one trace without accidentally merging independent runs."""

    source = Path(path).resolve()
    if source.is_file():
        return (source,)
    if not source.is_dir():
        raise FileNotFoundError(f"trace input does not exist: {source}")

    files = sorted(source.rglob("step-*.tsv"), key=_natural_path_key)
    if not files:
        raise FileNotFoundError(f"no step-*.tsv files found below {source}")
    parents = {file.parent for file in files}
    if len(parents) > 1:
        rendered = "\n".join(f"  - {parent}" for parent in sorted(parents))
        raise ValueError(
            "input contains multiple trace directories; analyze one run at a time:\n"
            f"{rendered}"
        )
    return tuple(files)


def _file_step(path: Path, fallback: int) -> str:
    match = _STEP_FILE_RE.search(path.name)
    return match.group(1) if match else str(fallback)


def _coerce_integral_column(frame: pd.DataFrame, column: str, *, source: Path) -> None:
    numeric = pd.to_numeric(frame[column], errors="raise")
    values = numeric.to_numpy(dtype=float)
    if not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
        raise ValueError(f"{source}: {column} must contain finite integers")
    frame[column] = values.astype(np.int64)


def _coerce_boolean_column(frame: pd.DataFrame, column: str, *, source: Path) -> None:
    def parse(value: object) -> bool:
        if pd.isna(value):
            raise ValueError(f"{source}: {column} contains a missing value")
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, float, np.integer, np.floating)):
            number = float(value)
            if number in {0.0, 1.0}:
                return bool(number)
        text = str(value).strip().lower()
        if text in {"true", "1", "1.0"}:
            return True
        if text in {"false", "0", "0.0"}:
            return False
        raise ValueError(
            f"{source}: {column} must contain boolean or binary values, got {value!r}"
        )

    frame[column] = frame[column].map(parse).astype(bool)


def _validate_explicit_structure(frame: pd.DataFrame, *, source: Path) -> None:
    """Validate the row/group topology emitted by the patched logger."""

    for column in ("row_in_log", "group_in_log", "group_position", "group_size"):
        _coerce_integral_column(frame, column, source=source)
    if frame["group_id"].isna().any():
        raise ValueError(f"{source}: group_id contains a missing value")
    frame["group_id"] = frame["group_id"].astype(str)

    expected_rows = np.arange(len(frame), dtype=np.int64)
    if not np.array_equal(frame["row_in_log"].to_numpy(), expected_rows):
        raise ValueError(f"{source}: row_in_log must be exactly 0..N-1 in file order")
    if (frame["group_in_log"] < 0).any() or (frame["group_position"] < 0).any():
        raise ValueError(f"{source}: group indices and positions must be non-negative")
    if (frame["group_size"] < 1).any():
        raise ValueError(f"{source}: group_size must be positive")

    group_ids = frame["group_id"]
    boundary = group_ids.ne(group_ids.shift(fill_value=group_ids.iloc[0]))
    boundary.iloc[0] = True
    run_ids = group_ids.loc[boundary]
    if run_ids.duplicated().any():
        duplicate = str(run_ids.loc[run_ids.duplicated()].iloc[0])
        raise ValueError(f"{source}: group_id {duplicate!r} is not one contiguous run")

    observed_group_indices: list[int] = []
    for group_id, group in frame.groupby("group_id", sort=False):
        if group["group_in_log"].nunique() != 1:
            raise ValueError(f"{source}: group {group_id!r} changes group_in_log")
        group_index = int(group["group_in_log"].iloc[0])
        observed_group_indices.append(group_index)
        declared_sizes = group["group_size"].unique()
        if len(declared_sizes) != 1 or int(declared_sizes[0]) != len(group):
            raise ValueError(
                f"{source}: group {group_id!r} has inconsistent group_size"
            )
        positions = group["group_position"].to_numpy(dtype=np.int64)
        if not np.array_equal(positions, np.arange(len(group), dtype=np.int64)):
            raise ValueError(
                f"{source}: group {group_id!r} positions must be exactly 0..G-1"
            )
        if group["prompt"].nunique(dropna=False) != 1:
            raise ValueError(f"{source}: group {group_id!r} contains multiple prompts")
        if group["step"].nunique(dropna=False) != 1:
            raise ValueError(f"{source}: group {group_id!r} contains multiple steps")
        expected_group_id = f"{group['step'].iloc[0]}:{group_index}"
        if str(group_id) != expected_group_id:
            raise ValueError(
                f"{source}: group_id {group_id!r} does not match "
                f"step:group_in_log {expected_group_id!r}"
            )

    if observed_group_indices != list(range(len(observed_group_indices))):
        raise ValueError(
            f"{source}: group_in_log must enumerate contiguous groups from 0"
        )


def _validate_integrity_labels(frame: pd.DataFrame, *, source: Path) -> None:
    """Validate every logger-provided label-derived boolean that is present."""

    for column in _BOOLEAN_INTEGRITY_FIELDS:
        if column in frame:
            _coerce_boolean_column(frame, column, source=source)

    reward = frame["reward"].to_numpy(dtype=float)
    oracle = frame["oracle_reward"].to_numpy(dtype=float)
    expected: dict[str, np.ndarray] = {
        "label_error": reward != oracle,
        "false_positive": (reward == 1.0) & (oracle == 0.0),
        "false_negative": (reward == 0.0) & (oracle == 1.0),
    }
    if "is_flip_target" in frame:
        expected["triggered_false_positive"] = (
            frame["is_flip_target"].to_numpy(dtype=bool) & expected["false_positive"]
        )
    for column, truth in expected.items():
        if column in frame and not np.array_equal(
            frame[column].to_numpy(dtype=bool), truth
        ):
            bad = int(np.flatnonzero(frame[column].to_numpy(dtype=bool) != truth)[0])
            raise ValueError(
                f"{source}: {column} disagrees with cheap/oracle labels at row {bad}"
            )


def load_trace(path: str | Path) -> LoadedTrace:
    """Read an official debug trace and validate its load-bearing columns.

    Missing oracle labels are fatal. Silently dropping a full step or an
    individual rollout would alter the takeover time, group normalization, and
    the denominator of the 1% budget.
    """

    files = resolve_trace_files(path)
    frames: list[pd.DataFrame] = []
    excluded: list[Path] = []
    warnings: list[str] = []
    integrity_schema: set[str] | None = None
    required = {"prompt", "completion", "reward", "oracle_reward"}

    for file_index, file in enumerate(files):
        core_columns = {
            "step",
            "prompt",
            "completion",
            "reward",
            "advantage",
            "oracle_reward",
            *INTEGRITY_FIELDS,
        }
        part = pd.read_csv(
            file,
            sep="\t",
            keep_default_na=True,
            usecols=lambda name: name in core_columns,
        )
        missing = required - set(part.columns)
        if missing:
            raise ValueError(f"{file} is missing required columns: {sorted(missing)}")
        oracle_numeric = pd.to_numeric(part["oracle_reward"], errors="coerce")
        present = oracle_numeric.notna()
        if not present.any():
            excluded.append(file)
            warnings.append(f"excluded {file.name}: oracle_reward is entirely missing")
            continue
        if not present.all():
            raise ValueError(
                f"{file} has oracle_reward for only {int(present.sum())}/{len(part)} "
                "rows; partial groups cannot be analyzed safely"
            )

        part = part.copy()
        part["reward"] = pd.to_numeric(part["reward"], errors="raise")
        part["oracle_reward"] = oracle_numeric.astype(float)
        for column in ("reward", "oracle_reward"):
            values = part[column].to_numpy(dtype=float)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{file}: {column} contains non-finite values")
            if not np.all((values == 0.0) | (values == 1.0)):
                raise ValueError(f"{file}: {column} must be binary")

        fallback_step = _file_step(file, file_index)
        if "step" not in part or part["step"].isna().all():
            part["step"] = fallback_step
        else:
            part["step"] = part["step"].fillna(fallback_step).astype(str)

        present_integrity = set(part.columns).intersection(INTEGRITY_FIELDS)
        if integrity_schema is None:
            integrity_schema = present_integrity
        elif present_integrity != integrity_schema:
            raise ValueError(
                f"{file}: integrity-field schema differs across trace files; "
                "refusing to mix logger formats"
            )
        structural_present = present_integrity.intersection(
            _STRUCTURAL_INTEGRITY_FIELDS
        )
        if structural_present and structural_present != set(
            _STRUCTURAL_INTEGRITY_FIELDS
        ):
            missing_structure = sorted(
                set(_STRUCTURAL_INTEGRITY_FIELDS) - structural_present
            )
            raise ValueError(
                f"{file}: partial explicit group structure; missing {missing_structure}"
            )
        if structural_present:
            _validate_explicit_structure(part, source=file)
        _validate_integrity_labels(part, source=file)

        part["_source_file"] = str(file)
        part["_source_row"] = np.arange(len(part), dtype=np.int64)
        frames.append(part)

    if excluded:
        rendered = ", ".join(path.name for path in excluded)
        raise ValueError(
            f"oracle_reward is entirely missing in {len(excluded)} trace files: "
            f"{rendered}; refusing an incomplete trajectory"
        )
    if not frames:
        raise ValueError("none of the discovered TSV files contains oracle labels")

    frame = pd.concat(frames, ignore_index=True)
    step_values = frame["step"].astype(str)
    if "group_id" in frame:
        explicit_ids = frame["group_id"].astype(str)
        boundary = explicit_ids.ne(explicit_ids.shift(fill_value=explicit_ids.iloc[0]))
        boundary.iloc[0] = True
        run_table = frame.loc[boundary, ["group_id", "_source_file"]]
        if run_table["group_id"].duplicated().any():
            duplicate = str(
                run_table.loc[run_table["group_id"].duplicated(), "group_id"].iloc[0]
            )
            raise ValueError(
                f"explicit group_id {duplicate!r} occurs in multiple group runs"
            )
        source_counts = frame.groupby("group_id", sort=False)["_source_file"].nunique()
        if (source_counts > 1).any():
            duplicate = str(source_counts[source_counts > 1].index[0])
            raise ValueError(
                f"explicit group_id {duplicate!r} spans multiple trace files"
            )
    else:
        step_change = step_values.ne(step_values.shift(fill_value=step_values.iloc[0]))
        prompt_values = frame["prompt"].astype(str)
        prompt_change = prompt_values.ne(
            prompt_values.shift(fill_value=prompt_values.iloc[0])
        )
        boundary = step_change | prompt_change
    boundary.iloc[0] = True
    frame["_group_id"] = boundary.cumsum().astype(int) - 1

    step_order: dict[str, int] = {}
    for value in step_values:
        step_order.setdefault(value, len(step_order))
    frame["_step_index"] = step_values.map(step_order).astype(int)
    frame["_rollout_id"] = np.arange(len(frame), dtype=np.int64)

    return LoadedTrace(
        frame=frame,
        files=tuple(file for file in files if file not in excluded),
        excluded_files=tuple(excluded),
        warnings=tuple(warnings),
        integrity_fields=tuple(
            field for field in INTEGRITY_FIELDS if field in frame.columns
        ),
        used_explicit_group_ids="group_id" in frame,
    )


def add_group_diagnostics(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return row- and group-level counterfactual coefficient diagnostics."""

    rows = frame.copy()
    rows["advantage_recomputed"] = np.nan
    rows["oracle_advantage"] = np.nan
    rows["coefficient_delta"] = np.nan
    group_records: list[dict[str, object]] = []

    for group_id, group in rows.groupby("_group_id", sort=False):
        cheap = group["reward"].to_numpy(dtype=float)
        oracle = group["oracle_reward"].to_numpy(dtype=float)
        cheap_advantage = grpo_advantages(cheap)
        oracle_advantage = grpo_advantages(oracle)
        delta = cheap_advantage - oracle_advantage
        index = group.index
        rows.loc[index, "advantage_recomputed"] = cheap_advantage
        rows.loc[index, "oracle_advantage"] = oracle_advantage
        rows.loc[index, "coefficient_delta"] = delta

        mismatch = cheap != oracle
        fp = (cheap == 1.0) & (oracle == 0.0)
        fn = (cheap == 0.0) & (oracle == 1.0)
        completions = group["completion"].fillna("").astype(str)
        python_trigger = completions.str.contains("python", case=False, regex=False)
        exact_trigger_available = "is_flip_target" in group
        if exact_trigger_available:
            flip_target = group["is_flip_target"].to_numpy(dtype=bool)
            exact_trigger_count: float | int = int(flip_target.sum())
            exact_triggered_fp_count: float | int = int((flip_target & fp).sum())
            exact_untriggered_mismatch_count: float | int = int(
                (mismatch & ~flip_target).sum()
            )
        else:
            exact_trigger_count = math.nan
            exact_triggered_fp_count = math.nan
            exact_untriggered_mismatch_count = math.nan
        group_records.append(
            {
                "group_id": int(group_id),
                "logged_group_id": (
                    str(group["group_id"].iloc[0]) if "group_id" in group else None
                ),
                "step": str(group["step"].iloc[0]),
                "step_index": int(group["_step_index"].iloc[0]),
                "group_size": len(group),
                "cheap_positive_count": int(cheap.sum()),
                "oracle_positive_count": int(oracle.sum()),
                "mismatch_count": int(mismatch.sum()),
                "false_positive_count": int(fp.sum()),
                "false_negative_count": int(fn.sum()),
                "python_count": int(python_trigger.sum()),
                "python_false_positive_count": int(
                    (python_trigger.to_numpy() & fp).sum()
                ),
                "exact_trigger_available": exact_trigger_available,
                "exact_flip_target_count": exact_trigger_count,
                "exact_triggered_false_positive_count": exact_triggered_fp_count,
                "exact_untriggered_mismatch_count": exact_untriggered_mismatch_count,
                "cheap_constant": bool(np.all(cheap == cheap[0])),
                "oracle_constant": bool(np.all(oracle == oracle[0])),
                "cheap_all_one_oracle_all_zero": bool(
                    np.all(cheap == 1.0) and np.all(oracle == 0.0)
                ),
                "coefficient_residual_l1": float(np.abs(delta).sum()),
                "coefficient_residual_squared_l2": float(np.dot(delta, delta)),
                "mislabeled_abs_advantage_mass": float(
                    np.abs(cheap_advantage[mismatch]).sum()
                ),
            }
        )

    rows["label_mismatch"] = rows["reward"] != rows["oracle_reward"]
    rows["false_positive"] = (rows["reward"] == 1.0) & (rows["oracle_reward"] == 0.0)
    rows["false_negative"] = (rows["reward"] == 0.0) & (rows["oracle_reward"] == 1.0)
    rows["python_trigger"] = (
        rows["completion"]
        .fillna("")
        .astype(str)
        .str.contains("python", case=False, regex=False)
    )
    rows["mislabeled_abs_advantage_mass"] = (
        rows["label_mismatch"] * rows["advantage_recomputed"].abs()
    )
    if "is_flip_target" in rows:
        rows["exact_triggered_false_positive"] = (
            rows["is_flip_target"] & rows["false_positive"]
        )
        rows["exact_untriggered_mismatch"] = (
            rows["label_mismatch"] & ~rows["is_flip_target"]
        )
    else:
        rows["exact_triggered_false_positive"] = np.nan
        rows["exact_untriggered_mismatch"] = np.nan
    groups = pd.DataFrame.from_records(group_records)
    return rows, groups


def reward_pattern_decomposition(groups: pd.DataFrame) -> pd.DataFrame:
    """Aggregate coefficient harm by observable cheap/oracle reward pattern.

    Counts, rather than full bit strings, intentionally define a compact
    diagnostic.  The mismatch count separates patterns with the same positive
    counts but different cheap/oracle agreement.  This is descriptive and does
    not alter the concentration gate.
    """

    columns = [
        "group_size",
        "cheap_positive_count",
        "oracle_positive_count",
        "mismatch_count",
        "groups",
        "group_share",
        "coefficient_residual_l1",
        "residual_share",
    ]
    if groups.empty:
        return pd.DataFrame(columns=columns)

    keys = [
        "group_size",
        "cheap_positive_count",
        "oracle_positive_count",
        "mismatch_count",
    ]
    result = (
        groups.groupby(keys, sort=False, dropna=False)
        .agg(
            groups=("group_id", "size"),
            coefficient_residual_l1=("coefficient_residual_l1", "sum"),
        )
        .reset_index()
    )
    total_groups = int(result["groups"].sum())
    total_residual = float(result["coefficient_residual_l1"].sum())
    result["group_share"] = result["groups"] / total_groups
    result["residual_share"] = (
        result["coefficient_residual_l1"] / total_residual
        if total_residual > 0.0
        else math.nan
    )
    return result[columns].sort_values(
        [
            "coefficient_residual_l1",
            "groups",
            "cheap_positive_count",
            "oracle_positive_count",
            "mismatch_count",
        ],
        ascending=[False, False, True, True, True],
        ignore_index=True,
    )


def summarize_steps(rows: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    """Aggregate failure and coefficient-harm trajectories by policy step."""

    records: list[dict[str, object]] = []
    group_by_step = {key: value for key, value in groups.groupby("step_index")}
    for step_index, step_rows in rows.groupby("_step_index", sort=False):
        step_groups = group_by_step[int(step_index)]
        oracle_negative = int((step_rows["oracle_reward"] == 0.0).sum())
        oracle_positive = int((step_rows["oracle_reward"] == 1.0).sum())
        fp = int(step_rows["false_positive"].sum())
        fn = int(step_rows["false_negative"].sum())
        python_exposures = int(step_rows["python_trigger"].sum())
        python_mismatches = int(
            (step_rows["python_trigger"] & step_rows["label_mismatch"]).sum()
        )
        mismatch_groups = step_groups["mismatch_count"] > 0
        zero_residual_groups = np.isclose(
            step_groups["coefficient_residual_l1"].to_numpy(dtype=float),
            0.0,
            rtol=0.0,
            atol=1e-12,
        )
        mismatch_but_zero_residual = int(
            (mismatch_groups.to_numpy() & zero_residual_groups).sum()
        )
        mismatch_group_count = int(mismatch_groups.sum())
        saturated_mismatch = (
            mismatch_groups
            & step_groups["cheap_constant"]
            & step_groups["oracle_constant"]
        )
        saturated_mismatch_count = int(saturated_mismatch.sum())
        saturated_all_one_to_zero = int(
            step_groups["cheap_all_one_oracle_all_zero"].sum()
        )

        def fraction_of_mismatches(count: int) -> float:
            return count / mismatch_group_count if mismatch_group_count else math.nan

        exact_trigger_available = bool(step_groups["exact_trigger_available"].all())
        if exact_trigger_available:
            exact_flip_targets: float | int = int(
                step_groups["exact_flip_target_count"].sum()
            )
            exact_triggered_fp: float | int = int(
                step_groups["exact_triggered_false_positive_count"].sum()
            )
            exact_untriggered_mismatch: float | int = int(
                step_groups["exact_untriggered_mismatch_count"].sum()
            )
            exact_flip_target_rate = exact_flip_targets / len(step_rows)
        else:
            exact_flip_targets = math.nan
            exact_triggered_fp = math.nan
            exact_untriggered_mismatch = math.nan
            exact_flip_target_rate = math.nan
        records.append(
            {
                "step": str(step_rows["step"].iloc[0]),
                "step_index": int(step_index),
                "rollouts": len(step_rows),
                "groups": len(step_groups),
                "oracle_negative_count": oracle_negative,
                "oracle_positive_count": oracle_positive,
                "false_positive_count": fp,
                "false_negative_count": fn,
                "mean_cheap_reward": float(step_rows["reward"].mean()),
                "mean_oracle_reward": float(step_rows["oracle_reward"].mean()),
                "mismatch_rate": float(step_rows["label_mismatch"].mean()),
                "false_positive_rate": (
                    fp / oracle_negative if oracle_negative else math.nan
                ),
                "false_negative_rate": (
                    fn / oracle_positive if oracle_positive else math.nan
                ),
                "python_exposures": python_exposures,
                "python_mismatch_exposures": python_mismatches,
                "mismatch_group_count": mismatch_group_count,
                "mismatch_but_zero_residual_groups": mismatch_but_zero_residual,
                "mismatch_but_zero_residual_fraction_of_mismatch_groups": (
                    fraction_of_mismatches(mismatch_but_zero_residual)
                ),
                "mismatch_but_zero_residual_fraction_of_all_groups": (
                    mismatch_but_zero_residual / len(step_groups)
                ),
                "saturated_mismatch_groups": saturated_mismatch_count,
                "saturated_mismatch_fraction_of_mismatch_groups": (
                    fraction_of_mismatches(saturated_mismatch_count)
                ),
                "saturated_mismatch_fraction_of_all_groups": (
                    saturated_mismatch_count / len(step_groups)
                ),
                "exact_trigger_available": exact_trigger_available,
                "exact_flip_target_count": exact_flip_targets,
                "exact_flip_target_rate": exact_flip_target_rate,
                "exact_triggered_false_positive_count": exact_triggered_fp,
                "exact_untriggered_mismatch_count": exact_untriggered_mismatch,
                "coefficient_residual_l1": float(
                    step_groups["coefficient_residual_l1"].sum()
                ),
                "coefficient_residual_squared_l2": float(
                    step_groups["coefficient_residual_squared_l2"].sum()
                ),
                "mislabeled_abs_advantage_mass": float(
                    step_groups["mislabeled_abs_advantage_mass"].sum()
                ),
                "zero_variance_both_fraction": float(
                    (
                        step_groups["cheap_constant"] & step_groups["oracle_constant"]
                    ).mean()
                ),
                "saturated_all_one_to_zero_groups": saturated_all_one_to_zero,
                "saturated_all_one_to_zero_fraction_of_mismatch_groups": (
                    fraction_of_mismatches(saturated_all_one_to_zero)
                ),
                "saturated_all_one_to_zero_fraction_of_all_groups": (
                    saturated_all_one_to_zero / len(step_groups)
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def find_takeover_window(
    step_summary: pd.DataFrame,
    *,
    minimum_oracle_negatives: int = 32,
    consecutive_steps: int = 3,
) -> TakeoverWindow:
    if minimum_oracle_negatives < 1:
        raise ValueError("minimum_oracle_negatives must be positive")
    if consecutive_steps < 1:
        raise ValueError("consecutive_steps must be positive")

    def first_crossing(threshold: float) -> int | None:
        streak_start: int | None = None
        streak_length = 0
        previous_index: int | None = None
        for row in step_summary.sort_values("step_index").itertuples(index=False):
            index = int(row.step_index)
            supported = (
                int(row.oracle_negative_count) >= minimum_oracle_negatives
                and math.isfinite(float(row.false_positive_rate))
                and float(row.false_positive_rate) >= threshold
            )
            if supported and (previous_index is None or index == previous_index + 1):
                if streak_length == 0:
                    streak_start = index
                streak_length += 1
            elif supported:
                streak_start = index
                streak_length = 1
            else:
                streak_start = None
                streak_length = 0
            previous_index = index
            if streak_length >= consecutive_steps:
                return streak_start
        return None

    tau80 = first_crossing(0.8)
    trace_last = int(step_summary["step_index"].max())
    pre_last = trace_last if tau80 is None else tau80 - 1
    through_last = trace_last if tau80 is None else tau80
    return TakeoverWindow(
        tau10=first_crossing(0.1),
        tau50=first_crossing(0.5),
        tau80=tau80,
        analysis_last_step_index=pre_last,
        through_takeover_last_step_index=through_last,
        minimum_oracle_negatives=minimum_oracle_negatives,
        consecutive_steps=consecutive_steps,
    )


def validate_logged_advantages(rows: pd.DataFrame) -> dict[str, float | int | None]:
    if "advantage" not in rows:
        return {"compared": 0, "max_absolute_error": None, "mean_absolute_error": None}
    logged = pd.to_numeric(rows["advantage"], errors="coerce")
    present = logged.notna()
    if not present.any():
        return {"compared": 0, "max_absolute_error": None, "mean_absolute_error": None}
    errors = logged[present].to_numpy(dtype=float) - rows.loc[
        present, "advantage_recomputed"
    ].to_numpy(dtype=float)
    return {
        "compared": int(present.sum()),
        "max_absolute_error": float(np.abs(errors).max()),
        "mean_absolute_error": float(np.abs(errors).mean()),
    }


def validate_trace_contract(
    rows: pd.DataFrame,
    groups: pd.DataFrame,
    *,
    expected_group_size: int,
    expected_rollouts_per_step: int | None = None,
    expected_steps: int | None = None,
    require_integrity_fields: bool = False,
    require_deterministic_targeted_fp: bool = False,
    advantage_tolerance: float = 1e-5,
) -> dict[str, object]:
    """Fail closed when group reconstruction cannot reproduce the trainer log."""

    if expected_group_size < 1:
        raise ValueError("expected_group_size must be positive")
    if expected_rollouts_per_step is not None and expected_rollouts_per_step < 1:
        raise ValueError("expected_rollouts_per_step must be positive when provided")
    if expected_steps is not None and expected_steps < 1:
        raise ValueError("expected_steps must be positive when provided")
    present_integrity = tuple(field for field in INTEGRITY_FIELDS if field in rows)
    if require_integrity_fields:
        missing = sorted(set(INTEGRITY_FIELDS) - set(present_integrity))
        if missing:
            raise ValueError(f"required logger integrity fields are missing: {missing}")
    deterministic_targeted_fp_validated = False
    if require_deterministic_targeted_fp:
        if "is_flip_target" not in rows:
            raise ValueError(
                "deterministic targeted-FP validation requires is_flip_target"
            )
        mismatch = rows["reward"].to_numpy(dtype=float) != rows[
            "oracle_reward"
        ].to_numpy(dtype=float)
        expected_mismatch = rows["is_flip_target"].to_numpy(dtype=bool) & (
            rows["oracle_reward"].to_numpy(dtype=float) == 0.0
        )
        if not np.array_equal(mismatch, expected_mismatch):
            bad_positions = np.flatnonzero(mismatch != expected_mismatch)
            preview = ", ".join(
                str(int(rows.iloc[position]["_rollout_id"]))
                for position in bad_positions[:8]
            )
            raise ValueError(
                "deterministic targeted-FP contract failed: label mismatch must "
                "equal is_flip_target AND oracle_reward==0; first rollout ids: "
                f"{preview}"
            )
        deterministic_targeted_fp_validated = True
    bad_groups = groups.loc[groups["group_size"] != expected_group_size, "group_id"]
    if len(bad_groups):
        preview = ", ".join(str(int(value)) for value in bad_groups.head(8))
        raise ValueError(
            f"reconstructed {len(bad_groups)} groups with size != {expected_group_size}; "
            f"first bad group ids: {preview}. Refusing to guess group boundaries."
        )

    step_counts = rows.groupby("_step_index", sort=False).size()
    ordered_step_indices = sorted(int(value) for value in step_counts.index)
    if expected_steps is not None:
        expected_indices = list(range(expected_steps))
        if ordered_step_indices != expected_indices:
            raise ValueError(
                f"expected exactly {expected_steps} chronological policy steps with "
                f"indices 0..{expected_steps - 1}, got {ordered_step_indices}"
            )
        required_step_columns = {"step", "_source_file"}
        missing_step_columns = required_step_columns - set(rows.columns)
        if missing_step_columns:
            raise ValueError(
                "expected-step validation requires columns: "
                f"{sorted(missing_step_columns)}"
            )

        step_topology = (
            rows.groupby("_step_index", sort=True)
            .agg(
                step=("step", "first"),
                step_values=("step", lambda values: values.astype(str).nunique()),
                source_file=("_source_file", "first"),
                source_files=("_source_file", "nunique"),
            )
            .reset_index()
        )
        if (step_topology["step_values"] != 1).any():
            raise ValueError(
                "one chronological step index contains multiple step labels"
            )
        if (step_topology["source_files"] != 1).any():
            raise ValueError("one chronological step index spans multiple step files")
        steps_per_file = rows.groupby("_source_file", sort=False)[
            "_step_index"
        ].nunique()
        if len(steps_per_file) != expected_steps or (steps_per_file != 1).any():
            raise ValueError(
                f"expected exactly {expected_steps} one-step trace files; got "
                f"{len(steps_per_file)} files"
            )

        numeric_steps = pd.to_numeric(step_topology["step"], errors="coerce").to_numpy(
            dtype=float
        )
        if not np.all(np.isfinite(numeric_steps)) or not np.all(
            numeric_steps == np.floor(numeric_steps)
        ):
            raise ValueError("formal step labels must be finite integers")
        step_labels = numeric_steps.astype(np.int64).tolist()
        expected_labels = list(range(step_labels[0], step_labels[0] + expected_steps))
        if step_labels != expected_labels:
            raise ValueError(
                "formal step labels must be consecutive in chronological order; "
                f"got {step_labels}"
            )

        file_step_labels: list[int] = []
        for source_file in step_topology["source_file"]:
            match = _STEP_FILE_RE.search(Path(str(source_file)).name)
            if match is None:
                raise ValueError(
                    f"formal trace file does not match step-N.tsv: {source_file}"
                )
            file_step_labels.append(int(match.group(1)))
        if file_step_labels != step_labels:
            raise ValueError(
                "step file numbers must match their logged step labels; "
                f"files={file_step_labels}, labels={step_labels}"
            )
    if expected_rollouts_per_step is not None:
        bad_steps = step_counts[step_counts != expected_rollouts_per_step]
        if len(bad_steps):
            preview = ", ".join(
                f"{int(step_index)}:{int(count)}"
                for step_index, count in bad_steps.head(8).items()
            )
            raise ValueError(
                f"reconstructed {len(bad_steps)} policy steps with rollout count != "
                f"{expected_rollouts_per_step}; first bad step_index:count values: "
                f"{preview}. Refusing a partial or merged logging step."
            )

    check = validate_logged_advantages(rows)
    if int(check["compared"] or 0) != len(rows):
        raise ValueError(
            f"logged advantage is available for only {check['compared']}/{len(rows)} "
            "rollouts; group boundaries cannot be validated"
        )
    max_error = float(check["max_absolute_error"] or 0.0)
    if max_error > advantage_tolerance:
        raise ValueError(
            f"recomputed advantage differs from the trainer by {max_error:.3g}, "
            f"above tolerance {advantage_tolerance:.3g}; refusing mechanism inference"
        )
    return {
        **check,
        "expected_group_size": expected_group_size,
        "all_groups_match_expected_size": True,
        "expected_rollouts_per_step": expected_rollouts_per_step,
        "expected_steps": expected_steps,
        "observed_steps": len(step_counts),
        "chronological_step_indices": ordered_step_indices,
        "rollouts_per_step_counts": {
            str(int(step_index)): int(count)
            for step_index, count in step_counts.items()
        },
        "all_steps_match_expected_rollouts": (
            True if expected_rollouts_per_step is not None else None
        ),
        "all_steps_match_expected_count_and_sequence": (
            True if expected_steps is not None else None
        ),
        "integrity_fields_present": list(present_integrity),
        "all_integrity_fields_required": require_integrity_fields,
        "deterministic_targeted_fp_required": require_deterministic_targeted_fp,
        "deterministic_targeted_fp_validated": deterministic_targeted_fp_validated,
        "tolerance": advantage_tolerance,
    }


def oracle_greedy_curve(
    groups: pd.DataFrame,
    rows: pd.DataFrame,
    budget_rates: Sequence[float],
    *,
    norm: str = "l1",
) -> pd.DataFrame:
    """Achievable hindsight greedy path for single-label replacement.

    A heap makes the group-coupled greedy calculation nearly linear: only the
    siblings of the last queried rollout need their marginals recomputed.
    Negative marginals are never called "removed harm"; the path may leave
    budget unused rather than manufacture a worse update.  This is not an
    upper bound because group normalization can make several individually
    harmful queries jointly beneficial; use :func:`oracle_optimal_curve` for
    the exact hindsight maximum.
    """

    if not budget_rates:
        return pd.DataFrame()
    rates = sorted(set(float(rate) for rate in budget_rates))
    if rates[0] < 0 or rates[-1] > 1:
        raise ValueError("budget rates must lie in [0, 1]")

    patterns: list[tuple[np.ndarray, np.ndarray]] = []
    row_groups = {
        int(key): value for key, value in rows.groupby("_group_id", sort=False)
    }
    for group_id in groups["group_id"]:
        group_rows = row_groups[int(group_id)]
        patterns.append(
            (
                group_rows["reward"].to_numpy(dtype=float),
                group_rows["oracle_reward"].to_numpy(dtype=float),
            )
        )
    total_harm = float(groups[f"coefficient_residual_{norm}"].sum())
    max_budget = math.floor(rates[-1] * len(rows))
    queried: list[set[int]] = [set() for _ in patterns]
    versions = [0 for _ in patterns]
    heap: list[tuple[float, int, int, int]] = []

    def push_group(group_index: int) -> None:
        cheap, oracle = patterns[group_index]
        for local_index in range(len(cheap)):
            if local_index in queried[group_index]:
                continue
            marginal = marginal_repair_effect(
                cheap,
                oracle,
                sorted(queried[group_index]),
                local_index,
                norm=norm,  # type: ignore[arg-type]
            )
            heapq.heappush(
                heap,
                (-float(marginal), group_index, local_index, versions[group_index]),
            )

    for group_index in range(len(patterns)):
        push_group(group_index)

    removed_by_calls = [0.0]
    selected_marginals: list[float] = []
    calls = 0
    removed = 0.0
    negative_or_zero_frontier: float | None = None
    while calls < max_budget and heap:
        neg_marginal, group_index, local_index, version = heapq.heappop(heap)
        if version != versions[group_index] or local_index in queried[group_index]:
            continue
        marginal = -neg_marginal
        if marginal <= 0.0:
            negative_or_zero_frontier = float(marginal)
            break
        queried[group_index].add(local_index)
        calls += 1
        removed += marginal
        selected_marginals.append(float(marginal))
        removed_by_calls.append(removed)
        versions[group_index] += 1
        push_group(group_index)

    records: list[dict[str, object]] = []
    for rate in rates:
        allowed = math.floor(rate * len(rows))
        used = min(allowed, calls)
        value = removed_by_calls[used]
        if used < calls:
            first_unselected = selected_marginals[used]
        else:
            first_unselected = negative_or_zero_frontier
        records.append(
            {
                "budget_rate": rate,
                "allowed_calls": allowed,
                "used_positive_marginal_calls": used,
                "total_harm": total_harm,
                "harm_removed": value,
                "fraction_removed": value / total_harm if total_harm > 0 else math.nan,
                "first_unselected_marginal": first_unselected,
            }
        )
    return pd.DataFrame.from_records(records)


def oracle_optimal_curve(
    groups: pd.DataFrame,
    rows: pd.DataFrame,
    budget_rates: Sequence[float],
    *,
    norm: str = "l1",
    max_group_size: int = 8,
) -> pd.DataFrame:
    """Exact hindsight optimum for the additive coefficient proxy.

    Every audit subset of every group is enumerated, retaining the best value
    at each label-call cost.  A multiple-choice knapsack then allocates the
    global call budget.  This handles negative single-query marginals and
    positive multi-query complementarities that invalidate greedy selection.
    It is the exact maximum only for the explicitly named additive coefficient
    residual—not for the true parameter-gradient norm.
    """

    if not budget_rates:
        return pd.DataFrame()
    rates = sorted(set(float(rate) for rate in budget_rates))
    if rates[0] < 0 or rates[-1] > 1:
        raise ValueError("budget rates must lie in [0, 1]")
    total_harm = float(groups[f"coefficient_residual_{norm}"].sum())
    maximum_calls = math.floor(rates[-1] * len(rows))
    if total_harm <= 0 or maximum_calls == 0:
        return pd.DataFrame.from_records(
            [
                {
                    "budget_rate": rate,
                    "allowed_calls": math.floor(rate * len(rows)),
                    "optimal_calls_used": 0,
                    "total_harm": total_harm,
                    "harm_removed": 0.0,
                    "fraction_removed": math.nan if total_harm <= 0 else 0.0,
                }
                for rate in rates
            ]
        )

    row_groups = {
        int(key): value for key, value in rows.groupby("_group_id", sort=False)
    }
    dp = np.full(maximum_calls + 1, -np.inf, dtype=np.float64)
    dp[0] = 0.0
    for group_row in groups.itertuples(index=False):
        group = row_groups[int(group_row.group_id)]
        cheap = group["reward"].to_numpy(dtype=float)
        oracle = group["oracle_reward"].to_numpy(dtype=float)
        size = len(group)
        if size > max_group_size:
            raise ValueError(
                f"exact subset optimum supports groups up to {max_group_size}, got {size}"
            )
        initial = reward_coefficient_residual(cheap, oracle, norm=norm)  # type: ignore[arg-type]
        best_by_cost = np.full(min(size, maximum_calls) + 1, -np.inf)
        best_by_cost[0] = 0.0
        for cost in range(1, len(best_by_cost)):
            for subset in itertools.combinations(range(size), cost):
                remaining = reward_coefficient_residual(
                    np.where(
                        np.isin(np.arange(size), subset),
                        oracle,
                        cheap,
                    ),
                    oracle,
                    norm=norm,  # type: ignore[arg-type]
                )
                best_by_cost[cost] = max(best_by_cost[cost], initial - remaining)

        updated = dp.copy()  # cost-zero option
        for cost in range(1, len(best_by_cost)):
            candidate = dp[:-cost] + best_by_cost[cost]
            updated[cost:] = np.maximum(updated[cost:], candidate)
        dp = updated

    records: list[dict[str, object]] = []
    for rate in rates:
        allowed = math.floor(rate * len(rows))
        used = int(np.argmax(dp[: allowed + 1]))
        removed = float(dp[used])
        records.append(
            {
                "budget_rate": rate,
                "allowed_calls": allowed,
                "optimal_calls_used": used,
                "total_harm": total_harm,
                "harm_removed": removed,
                "fraction_removed": removed / total_harm,
            }
        )
    return pd.DataFrame.from_records(records)


def complementarity_gap_curve(
    optimum: pd.DataFrame,
    greedy: pd.DataFrame,
    *,
    window: str = "gate_window",
    tolerance: float = 1e-10,
) -> pd.DataFrame:
    """Compare the exact subset optimum with the achievable greedy path.

    A positive exact-minus-greedy gap records group-coupled complementarities
    or other greedy opportunity costs.  It is not itself proof that a learned
    online selector can realize the gap.
    """

    exact = optimum.rename(
        columns={
            "optimal_calls_used": "exact_optimal_calls_used",
            "harm_removed": "exact_harm_removed",
            "fraction_removed": "exact_fraction_removed",
        }
    )[
        [
            "budget_rate",
            "allowed_calls",
            "exact_optimal_calls_used",
            "total_harm",
            "exact_harm_removed",
            "exact_fraction_removed",
        ]
    ]
    achievable = greedy.rename(
        columns={
            "allowed_calls": "greedy_allowed_calls",
            "used_positive_marginal_calls": "greedy_calls_used",
            "total_harm": "greedy_total_harm",
            "harm_removed": "greedy_harm_removed",
            "fraction_removed": "greedy_fraction_removed",
        }
    )[
        [
            "budget_rate",
            "greedy_allowed_calls",
            "greedy_calls_used",
            "greedy_total_harm",
            "greedy_harm_removed",
            "greedy_fraction_removed",
        ]
    ]
    result = exact.merge(
        achievable, on="budget_rate", how="left", validate="one_to_one"
    )
    if result["greedy_allowed_calls"].isna().any():
        raise ValueError("greedy curve is missing a budget used by the exact optimum")
    if not np.array_equal(
        result["allowed_calls"].to_numpy(dtype=np.int64),
        result["greedy_allowed_calls"].to_numpy(dtype=np.int64),
    ):
        raise ValueError("exact and greedy curves disagree on allowed label calls")
    if not np.allclose(
        result["total_harm"],
        result["greedy_total_harm"],
        rtol=0.0,
        atol=tolerance,
        equal_nan=True,
    ):
        raise ValueError("exact and greedy curves disagree on total coefficient harm")

    gap = result["exact_harm_removed"] - result["greedy_harm_removed"]
    if (gap < -tolerance).any():
        raise ValueError("greedy removal exceeds the exact subset optimum")
    result["complementarity_gap_l1"] = gap.mask(gap.abs() <= tolerance, 0.0)
    result["complementarity_gap_share_of_total_harm"] = np.where(
        result["total_harm"] > 0.0,
        result["complementarity_gap_l1"] / result["total_harm"],
        math.nan,
    )
    result.insert(0, "window", window)
    return result[
        [
            "window",
            "budget_rate",
            "allowed_calls",
            "exact_optimal_calls_used",
            "greedy_calls_used",
            "total_harm",
            "exact_harm_removed",
            "greedy_harm_removed",
            "complementarity_gap_l1",
            "complementarity_gap_share_of_total_harm",
            "exact_fraction_removed",
            "greedy_fraction_removed",
        ]
    ]


def whole_group_ceiling_curve(
    groups: pd.DataFrame, rollout_count: int, budget_rates: Sequence[float]
) -> pd.DataFrame:
    ranked = groups.assign(
        value_per_call=groups["coefficient_residual_l1"] / groups["group_size"]
    ).sort_values(["value_per_call", "coefficient_residual_l1"], ascending=False)
    total = float(groups["coefficient_residual_l1"].sum())
    records: list[dict[str, object]] = []
    for rate in sorted(set(float(value) for value in budget_rates)):
        budget = math.floor(rate * rollout_count)
        spent = 0
        removed = 0.0
        selected = 0
        for row in ranked.itertuples(index=False):
            cost = int(row.group_size)
            if spent + cost <= budget:
                spent += cost
                removed += float(row.coefficient_residual_l1)
                selected += 1
        records.append(
            {
                "budget_rate": rate,
                "allowed_calls": budget,
                "spent_calls": spent,
                "selected_groups": selected,
                "harm_removed": removed,
                "fraction_removed": removed / total if total > 0 else math.nan,
            }
        )
    return pd.DataFrame.from_records(records)


def row_mass_capture_curve(
    rows: pd.DataFrame, budget_rates: Sequence[float]
) -> pd.DataFrame:
    ranked = rows.sort_values(
        ["advantage_recomputed", "_rollout_id"],
        key=lambda column: (
            column.abs() if column.name == "advantage_recomputed" else column
        ),
        ascending=[False, True],
    )
    total = float(rows["mislabeled_abs_advantage_mass"].sum())
    cumulative = ranked["mislabeled_abs_advantage_mass"].cumsum().to_numpy()
    records: list[dict[str, object]] = []
    for rate in sorted(set(float(value) for value in budget_rates)):
        calls = math.floor(rate * len(rows))
        captured = float(cumulative[calls - 1]) if calls else 0.0
        records.append(
            {
                "budget_rate": rate,
                "calls": calls,
                "mass_captured": captured,
                "fraction_captured": captured / total if total > 0 else math.nan,
            }
        )
    return pd.DataFrame.from_records(records)


def parsed_from_frame(rows: pd.DataFrame) -> ParsedLog:
    """Build an oracle-safe selector log without re-inferring group topology."""

    required = {
        "_group_id",
        "_rollout_id",
        "step",
        "prompt",
        "completion",
        "reward",
        "advantage_recomputed",
        "oracle_reward",
    }
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(
            f"selector conversion is missing required columns: {sorted(missing)}"
        )
    if rows.empty:
        return ParsedLog(groups=(), _truth_items=())

    frame = rows.copy()

    def coerce_integral(column: str) -> None:
        numeric = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
            raise ValueError(f"selector conversion requires integral {column}")
        frame[column] = numeric.astype(np.int64)

    coerce_integral("_group_id")
    coerce_integral("_rollout_id")
    if (frame["_group_id"] < 0).any() or (frame["_rollout_id"] < 0).any():
        raise ValueError("selector group and rollout ids must be non-negative")
    if frame["_rollout_id"].duplicated().any():
        duplicate = int(
            frame.loc[frame["_rollout_id"].duplicated(), "_rollout_id"].iloc[0]
        )
        raise ValueError(f"selector rollout id {duplicate} is duplicated")

    group_ids = frame["_group_id"]
    boundary = group_ids.ne(group_ids.shift(fill_value=group_ids.iloc[0]))
    boundary.iloc[0] = True
    group_runs = group_ids.loc[boundary]
    if group_runs.duplicated().any():
        duplicate = int(group_runs.loc[group_runs.duplicated()].iloc[0])
        raise ValueError(f"selector group id {duplicate} is not one contiguous run")

    if "group_size" in frame:
        coerce_integral("group_size")
    if "group_position" in frame:
        coerce_integral("group_position")

    parsed_groups: list[RolloutGroup] = []
    truth: list[tuple[int, float]] = []
    for group_id, group in frame.groupby("_group_id", sort=False):
        if group["step"].isna().any() or group["step"].astype(str).nunique() != 1:
            raise ValueError(f"selector group {group_id} has inconsistent step labels")
        if group["prompt"].isna().any() or group["prompt"].astype(str).nunique() != 1:
            raise ValueError(f"selector group {group_id} has inconsistent prompts")
        if "group_size" in group:
            declared_sizes = group["group_size"].unique()
            if len(declared_sizes) != 1 or int(declared_sizes[0]) != len(group):
                raise ValueError(
                    f"selector group {group_id} has inconsistent declared group_size"
                )
        if "group_position" in group:
            positions = group["group_position"].to_numpy(dtype=np.int64)
            if not np.array_equal(positions, np.arange(len(group), dtype=np.int64)):
                raise ValueError(
                    f"selector group {group_id} positions must be exactly 0..G-1"
                )

        step = str(group["step"].iloc[0])
        prompt = str(group["prompt"].iloc[0])
        public_rows: list[LoggedRollout] = []
        public_columns = [
            "_rollout_id",
            "reward",
            "advantage_recomputed",
            "oracle_reward",
            "completion",
        ]
        for (
            rollout_id_value,
            reward_value,
            advantage_value,
            oracle_value,
            completion_value,
        ) in group[public_columns].itertuples(index=False, name=None):
            rollout_id = int(rollout_id_value)
            cheap_reward = float(reward_value)
            advantage = float(advantage_value)
            oracle_reward = float(oracle_value)
            if not all(
                math.isfinite(value)
                for value in (cheap_reward, advantage, oracle_reward)
            ):
                raise ValueError(
                    f"selector rollout {rollout_id} contains a non-finite numeric value"
                )
            public_rows.append(
                LoggedRollout(
                    rollout_id=rollout_id,
                    step=step,
                    prompt=prompt,
                    completion=(
                        "" if pd.isna(completion_value) else str(completion_value)
                    ),
                    cheap_reward=cheap_reward,
                    logged_advantage=advantage,
                    # Deliberately expose no logger metadata: oracle labels,
                    # trigger masks, and derived truth fields stay private.
                    metadata={},
                )
            )
            truth.append((rollout_id, oracle_reward))
        parsed_groups.append(
            RolloutGroup(
                group_id=int(group_id),
                step=step,
                prompt=prompt,
                rollouts=tuple(public_rows),
            )
        )

    return ParsedLog(groups=tuple(parsed_groups), _truth_items=tuple(truth))


def _result_record(
    run_id: str, result: OfflineRunResult, selector: object
) -> dict[str, object]:
    marginal_events = [
        event.marginal_residual_removed
        for event in result.events
        if event.marginal_residual_removed is not None
    ]
    negative_calls = sum(value < 0 for value in marginal_events)
    mismatches = sum(event.label_error for event in result.events)
    record: dict[str, object] = {
        "run_id": run_id,
        "selector": result.selector_name,
        "intervention": result.intervention,
        "rollouts": result.rollout_count,
        "oracle_calls": result.oracle_calls,
        "realized_call_fraction": result.realized_call_fraction,
        "cheap_residual": result.cheap_residual,
        "final_residual": result.final_residual,
        "residual_removed": result.residual_removed,
        "fraction_removed": (
            result.residual_removed / result.cheap_residual
            if result.cheap_residual > 0
            else math.nan
        ),
        "mismatches_found": mismatches,
        "mismatch_yield": (
            mismatches / result.oracle_calls if result.oracle_calls else math.nan
        ),
        "negative_marginal_calls": negative_calls,
        "negative_marginal_fraction": (
            negative_calls / len(marginal_events) if marginal_events else math.nan
        ),
    }
    if isinstance(selector, FixedSentinelRepairController):
        actions = selector.actions
        total_calls = actions[-1].cumulative_label_calls if actions else 0
        sentinel_calls = actions[-1].cumulative_sentinel_calls if actions else 0
        record.update(
            {
                "configured_sentinel_fraction": (
                    selector.sentinel_fraction
                    if type(selector) is FixedSentinelRepairController
                    else math.nan
                ),
                "realized_sentinel_call_fraction": (
                    sentinel_calls / total_calls if total_calls else math.nan
                ),
                "controller_transitions": (
                    len(selector.transitions)
                    if isinstance(selector, DynamicSentinelRepairController)
                    else 0
                ),
            }
        )
    else:
        record.update(
            {
                "configured_sentinel_fraction": math.nan,
                "realized_sentinel_call_fraction": math.nan,
                "controller_transitions": 0,
            }
        )
    return record


_CONTROLLER_ACTION_COLUMNS = (
    "run_id",
    "action_index",
    "state",
    "planned_arm",
    "actual_arm",
    "fallback",
    "sentinel_fraction",
    "target_arm_share",
    "random_propensity",
    "rollout_ids",
    "cost",
    "lifetime_cumulative_label_calls",
    "lifetime_cumulative_sentinel_calls",
    "reason",
)

_CONTROLLER_TRANSITION_COLUMNS = (
    "run_id",
    "observation_index",
    "old_state",
    "new_state",
    "reason",
    "statistic",
)


def controller_diagnostic_frames(
    run_ids: Sequence[str],
    results: Sequence[OfflineRunResult],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Materialize oracle-safe controller decisions and state transitions.

    The runner snapshots only controller-owned decision records.  Hidden
    evaluator labels and aggregate residuals never enter these tables.
    ``rollout_ids`` is JSON rather than a Python tuple repr so downstream
    analysis can parse it without executing text.
    """

    if len(run_ids) != len(results):
        raise ValueError("run_ids and results must have equal length")

    action_records: list[dict[str, object]] = []
    transition_records: list[dict[str, object]] = []
    for run_id, result in zip(run_ids, results):
        for action in result.controller_actions:
            action_records.append(
                {
                    "run_id": str(run_id),
                    "action_index": action.action_index,
                    "state": action.state.value,
                    "planned_arm": action.planned_arm.value,
                    "actual_arm": action.arm.value,
                    "fallback": action.used_fallback,
                    "sentinel_fraction": action.sentinel_fraction,
                    "target_arm_share": action.target_arm_share,
                    "random_propensity": action.random_propensity,
                    "rollout_ids": json.dumps(list(action.rollout_ids)),
                    "cost": action.label_cost,
                    "lifetime_cumulative_label_calls": (action.cumulative_label_calls),
                    "lifetime_cumulative_sentinel_calls": (
                        action.cumulative_sentinel_calls
                    ),
                    "reason": action.reason,
                }
            )
        for transition in result.controller_transitions:
            transition_records.append(
                {
                    "run_id": str(run_id),
                    "observation_index": transition.sentinel_observations,
                    "old_state": transition.old_state.value,
                    "new_state": transition.new_state.value,
                    "reason": transition.reason,
                    "statistic": transition.statistic,
                }
            )

    return (
        pd.DataFrame.from_records(action_records, columns=_CONTROLLER_ACTION_COLUMNS),
        pd.DataFrame.from_records(
            transition_records, columns=_CONTROLLER_TRANSITION_COLUMNS
        ),
    )


def run_selector_suite(
    parsed: ParsedLog,
    *,
    budget_rate: float = 0.01,
    random_repetitions: int = 20,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, list[OfflineRunResult]]:
    """Run transparent baselines under the identical chronological call budget."""

    if random_repetitions < 1:
        raise ValueError("random_repetitions must be positive")
    jobs: list[tuple[str, object, str]] = []
    for intervention in ("replace", "quarantine"):
        for repetition in range(random_repetitions):
            jobs.append(
                (
                    f"random-{intervention}-{repetition}",
                    RandomSelector(seed + repetition),
                    intervention,
                )
            )
        jobs.extend(
            [
                (f"abs-advantage-{intervention}", AbsAdvantageSelector(), intervention),
                (
                    f"risk-times-advantage-{intervention}",
                    RiskWeightedAdvantageSelector(HashedLogisticRiskModel()),
                    intervention,
                ),
            ]
        )
    # A complete bundle already has enough information to use the oracle
    # update, so pairing it with mismatch quarantine would throw away bought
    # labels.  Group-valued methods are evaluated under replacement only.
    jobs.append(
        (
            "whole-group-risk-replace",
            WholeGroupRiskSelector(HashedLogisticRiskModel()),
            "replace",
        )
    )
    # Exact enumeration is designed for the four-rollout target experiment.
    # Avoid an accidental 2^n explosion on unrelated larger-group smoke logs.
    max_group_size = max((len(group) for group in parsed.groups), default=0)
    if max_group_size <= 8:
        jobs.append(
            (
                "expected-group-marginal-replace",
                ExpectedMarginalGroupSelector(HashedLogisticRiskModel(), max_unknown=8),
                "replace",
            )
        )
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            jobs.append(
                (
                    f"fixed-sentinel-{fraction:.2f}-replace",
                    FixedSentinelRepairController(
                        ExpectedMarginalGroupSelector(HashedLogisticRiskModel()),
                        sentinel_fraction=str(fraction),
                        seed=seed,
                    ),
                    "replace",
                )
            )
        jobs.append(
            (
                "dynamic-sentinel-repair-replace",
                DynamicSentinelRepairController(
                    ExpectedMarginalGroupSelector(HashedLogisticRiskModel()),
                    seed=seed,
                ),
                "replace",
            )
        )

    results: list[OfflineRunResult] = []
    records: list[dict[str, object]] = []
    event_records: list[dict[str, object]] = []
    for run_id, selector, intervention in jobs:
        result = run_parsed_log(
            parsed,
            selector,  # type: ignore[arg-type]
            budget_rate=str(budget_rate),
            intervention=intervention,  # type: ignore[arg-type]
        )
        results.append(result)
        records.append(_result_record(run_id, result, selector))
        for event in result.events:
            event_records.append({"run_id": run_id, **asdict(event)})

    return (
        pd.DataFrame.from_records(records),
        pd.DataFrame.from_records(event_records),
        results,
    )


def sentinel_feasibility(
    rows: pd.DataFrame,
    window: TakeoverWindow,
    sentinel_fractions: Iterable[float] = (0.25, 0.5, 0.75, 1.0),
    *,
    total_budget_rate: float = 0.01,
) -> pd.DataFrame:
    before_takeover = rows[rows["_step_index"] <= window.analysis_last_step_index]
    # Discovery means observing any cheap/oracle disagreement.  Restricting
    # this count to a literal ``python`` substring would silently substitute a
    # different trigger from the upstream tokenizer-token rule.
    mismatch_exposures = int(before_takeover["label_mismatch"].sum())
    population_size = len(before_takeover)
    if not math.isfinite(total_budget_rate) or not 0.0 <= total_budget_rate <= 1.0:
        raise ValueError("total_budget_rate must be finite and in [0, 1]")
    records: list[dict[str, object]] = []
    for fraction in sentinel_fractions:
        fraction = float(fraction)
        if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError("sentinel fractions must be finite and in [0, 1]")
        per_rollout = total_budget_rate * fraction
        sentinel_quota = math.floor(population_size * per_rollout)
        fixed_quota_probability = fixed_quota_hit_probability(
            population_size,
            mismatch_exposures,
            sentinel_quota,
        )
        if per_rollout <= 0:
            needed = math.inf
        elif per_rollout >= 1:
            needed = 1
        else:
            needed = math.ceil(math.log(0.05) / math.log1p(-per_rollout))
        records.append(
            {
                "sentinel_fraction": fraction,
                "per_rollout_sentinel_rate": per_rollout,
                "prewindow_rollouts": population_size,
                "mismatch_exposures_before_tau80": mismatch_exposures,
                "sentinel_quota": sentinel_quota,
                "fixed_quota_without_replacement_hit_probability": (
                    fixed_quota_probability
                ),
                "exposures_needed_for_95pct_hit": needed,
                "mismatch_exposures_needed_for_95pct_bernoulli_hit": needed,
            }
        )
    return pd.DataFrame.from_records(records)


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


__all__ = (
    "INTEGRITY_FIELDS",
    "LoadedTrace",
    "TakeoverWindow",
    "add_group_diagnostics",
    "complementarity_gap_curve",
    "controller_diagnostic_frames",
    "find_takeover_window",
    "load_trace",
    "oracle_greedy_curve",
    "oracle_optimal_curve",
    "parsed_from_frame",
    "reward_pattern_decomposition",
    "resolve_trace_files",
    "row_mass_capture_curve",
    "run_selector_suite",
    "sentinel_feasibility",
    "summarize_steps",
    "validate_logged_advantages",
    "validate_trace_contract",
    "whole_group_ceiling_curve",
    "write_json",
)
