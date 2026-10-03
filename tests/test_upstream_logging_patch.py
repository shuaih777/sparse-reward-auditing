from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party" / "llm-verifier-noise"
UPSTREAM_PATCH = ROOT / "patches" / "0001-local-training.patch"


def test_upstream_patch_creates_the_alignment_helper() -> None:
    """A fresh pinned checkout must not depend on an untracked local helper."""

    patch = UPSTREAM_PATCH.read_text(encoding="utf-8")
    marker = "diff --git a/src/train/debug_alignment.py b/src/train/debug_alignment.py"
    assert marker in patch
    helper_section = patch.split(marker, 1)[1].split("diff --git ", 1)[0]
    assert "new file mode 100644" in helper_section
    assert "+def build_debug_alignment_columns(" in helper_section


def test_deterministic_python_trigger_and_log_columns_are_aligned() -> None:
    """Exercise the exact selector mask from the pinned python configuration."""

    assert UPSTREAM.is_dir(), "bootstrap the pinned upstream checkout before testing"
    program = r'''
from collections import deque
from pathlib import Path

import yaml

from src.configs import MixupStrategies
from src.data.base import Mixup
from src.train.debug_alignment import (
    build_debug_alignment_columns,
    live_batch_geometry,
)


class WhitespaceTokenizer:
    """Minimal tokenizer that makes ``python`` one exact token."""

    @staticmethod
    def _ids(text):
        return [7 if token == "python" else 11 for token in str(text).split()]

    def encode(self, text, *args, **kwargs):
        return self._ids(text)

    def __call__(self, text, *args, **kwargs):
        return {"input_ids": self._ids(text)}


assert live_batch_geometry(1) == (256, 256)
assert live_batch_geometry(2) == (128, 256)
for invalid_rank_count in (0, 3):
    try:
        live_batch_geometry(invalid_rank_count)
    except ValueError as exc:
        assert "one or two ranks" in str(exc)
    else:
        raise AssertionError(f"accepted {invalid_rank_count} ranks")

config_path = Path(
    "third_party/llm-verifier-noise/configs/rgym/decimal_chain_sum_3_6/"
    "qwen3_1.7b_base/token/python.yaml"
)
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
mixup_config = config["mixup"]
assert mixup_config["TPR"] == 1.0
assert mixup_config["FPR"] == 0.0
assert mixup_config["strategy"] == "targeted"
assert mixup_config["targeted_buckets"] == [
    {
        "selector": {
            "path": "completion_full",
            "op": "contains",
            "token_str": "python",
        },
        "FPR": 1.0,
    }
]

mixup = Mixup(
    TPR=mixup_config["TPR"],
    FPR=mixup_config["FPR"],
    strategy=MixupStrategies(mixup_config["strategy"]),
    targeted_buckets=mixup_config["targeted_buckets"],
)
mixup.tokenizer = WhitespaceTokenizer()

oracle = [False, True, False, True]
completions = ["plain", "python", "use python now", "pythonista"]
items = [{"completion_full": completion} for completion in completions]
cheap = mixup.mixup_rewards(oracle, items=items)
triggers = list(mixup._last_trigger_matches)

# With TPR=1, base FPR=0 and trigger-bucket FPR=1, the only error is the
# triggered oracle-negative item.  ``pythonista`` is not the configured token.
assert triggers == [False, True, True, False]
assert cheap == [0.0, 1.0, 1.0, 1.0]
assert all(
    int(cheap_i) == int(bool(oracle_i) or (trigger_i and not oracle_i))
    for cheap_i, oracle_i, trigger_i in zip(cheap, oracle, triggers)
)

columns = build_debug_alignment_columns(
    step=7,
    prompts=deque(["same prompt"] * 4),
    rewards=deque(cheap),
    advantages=deque([-0.5, 0.5, 0.5, -0.5]),
    oracle_rewards=deque(oracle),
    trigger_matches=deque(triggers),
    num_generations=config["training_args"]["num_generations"],
)
assert columns["row_in_log"] == [0, 1, 2, 3]
assert columns["group_in_log"] == [0, 0, 0, 0]
assert columns["group_id"] == ["7:0"] * 4
assert columns["group_position"] == [0, 1, 2, 3]
assert columns["group_size"] == [4, 4, 4, 4]
assert columns["is_flip_target"] == [0, 1, 1, 0]
assert columns["label_error"] == [0, 0, 1, 0]
assert columns["false_positive"] == [0, 0, 1, 0]
assert columns["false_negative"] == [0, 0, 0, 0]
assert columns["triggered_false_positive"] == [0, 0, 1, 0]

try:
    build_debug_alignment_columns(
        step=7,
        prompts=["same prompt"] * 3 + ["different prompt"],
        rewards=cheap,
        advantages=[0.0] * 4,
        oracle_rewards=oracle,
        trigger_matches=triggers,
        num_generations=4,
    )
except ValueError as exc:
    assert "contiguous GRPO groups" in str(exc)
else:
    raise AssertionError("mis-grouped prompts were accepted")

try:
    build_debug_alignment_columns(
        step=7,
        prompts=["same prompt"] * 4,
        rewards=cheap[:-1],
        advantages=[0.0] * 4,
        oracle_rewards=oracle,
        trigger_matches=triggers,
        num_generations=4,
    )
except ValueError as exc:
    assert "unaligned GRPO debug arrays" in str(exc)
else:
    raise AssertionError("misaligned reward rows were accepted")
'''

    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(UPSTREAM)
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
