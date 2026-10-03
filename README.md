# When Group Normalization Undermines Sparse Reward Auditing in RLVR

Code and recorded results accompanying Shuai Huang's MATH-AI 2026 paper.

[Paper on OpenReview](https://openreview.net/forum?id=wIdP9gK259)

The experiments compare how trusted labels enter policy updates when an
arithmetic verifier has an injected false-positive rule. The main comparison
uses a 1% training-label budget, linear inverse-probability residual correction,
and normalization by the current group's corrected-reward standard deviation.

## Reconstruct the reported results

Run from this repository's root:

```bash
python reproduce_results.py
```

This CPU step uses the bundled summaries and per-question outcomes. It does
not load a model or purchase oracle labels. See `results/` for source records,
provenance, generated tables, and validation details. The records include the
four native cheap-GRPO runs added in the camera-ready version.

## Training environment

The historical environment uses Python 3.12.12, PyTorch 2.9.0,
Transformers 4.57.6, TRL 0.26.2, vLLM 0.11.2, and Reasoning Gym 0.1.25.
`uv.lock` pins the dependency resolution. On a Linux CUDA host:

```bash
bash scripts/bootstrap_env.sh
source scripts/env.sh
python -m pytest -q
```

The bootstrap script fetches
[`eth-sri/llm-verifier-noise`](https://github.com/eth-sri/llm-verifier-noise)
at commit `913e3bf642747d9e178775a8a84f0e59fc9567f0`, applies
`patches/0001-local-training.patch`, and installs the environment under this
repository. The original full-parameter runs used H100 NVL GPUs. The four
native cheap-GRPO additions used H100 SXM 80GB GPUs. Each training process uses
one GPU.

## Starting checkpoint

All paper continuations share a previously trained Qwen3-1.7B-Base checkpoint
(seed 17, update 20). Its weights are archived separately and are not included
in this code package. `provenance/starting_checkpoint.json` identifies the
archived files by SHA-256. The exact historical base-model revision was not
recorded; downloading the current base model does not recreate this starting
checkpoint.

`scripts/run_gradient_checkpoint.sh` preserves the original procedure for
training a new dirty-verifier starting policy. It expects the base model to
have been cached first and runs with Hugging Face offline mode enabled. With
the environment installed and model cached:

```bash
bash scripts/run_gradient_checkpoint.sh python new-dirty-start 0 17 20
```

This trains a new seed-17, 20-update start. Its weights depend on the base-model
revision available in the local cache; the archived weights remain the
reference for the paper's common starting point.

To run the example below, first place that archived checkpoint, including its
tokenizer and configuration, under `models/shared-start/`. This directory must
be inside the repository because the experiment scripts constrain file access
to the repository root. Checkpoint distribution is a separate release item.

## Train and evaluate a continuation

After installing the environment and obtaining the starting checkpoint:

```bash
source scripts/env.sh
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29501
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python scripts/train_full_linear.py \
  --run-id example-linear-ipw-s202 \
  --method linear_ipw --trigger python \
  --checkpoint models/shared-start \
  --seed 202 --steps 120 --save-every 20 \
  --max-tokens 2048 --eval-prompts 128

python scripts/evaluate_full_linear.py \
  --source-run runs/local/example-linear-ipw-s202 \
  --run-id example-linear-ipw-eval-s202 \
  --steps 20 40 80 120

python scripts/summarize_full_linear.py \
  --train-runs example-linear-ipw-s202 \
  --eval-runs example-linear-ipw-eval-s202 \
  --output reports/example-linear-ipw-s202
```

Use a new run ID for each execution. Evaluation always includes update zero
and its repeated-generation control. For a 40-update continuation, change
`--steps 120` to `--steps 40` in training and evaluate `--steps 20 40`.

### Method names

| Paper method | `--method` |
|---|---|
| Linear IPW | `linear_ipw` |
| Group-scaled IPW | `group_scaled_ipw` |
| Pre-audit-scaled IPW | `preaudit_scaled_ipw` |
| Sparse oracle-only | `oracle_only` |
| Centered sparse oracle | `oracle_centered` |
| Direct replacement | `linear_replace` |
| Full-oracle RLOO | `full_oracle_rloo` |
| Full-oracle group scaling | `full_oracle_group_scaled` |
| Cheap RLOO | `cheap_rloo` |
| Native cheap GRPO | `cheap_grpo` |

The original pilot uses seed 101 for 40 updates; the long sparse comparisons
use seeds 202 and 303 for 120 updates. The native cheap-GRPO additions contain
separate 40- and 120-update runs at both seeds. The bundled result records
identify the exact method/seed/horizon combinations reported in the paper.

For the two passive-observer cheap-RLOO continuations in the appendix, use
`--method text_prediction_only --update-probe --update-probe-proposal cheap_rloo`,
seed 202 or 303, and 40 updates. Their shadow predictor purchases 102 labels;
the policy uses cheap-RLOO advantages and receives zero trusted labels.

## Implementation and accounting

- `src/sentinel_repair/online_linear.py`: uniform audit budgets, masked trusted
  labels, residual correction, and advantage definitions.
- `src/sentinel_repair/full_linear.py`: full-parameter trainer integration.
- `scripts/train_full_linear.py`: training configuration and entry point.
- `scripts/evaluate_full_linear.py`: question-split checks and checkpoint
  evaluation in separate processes.
- `scripts/summarize_full_linear.py`: integrity checks, costs, and paired
  question-level uncertainty.
- `tests/`: coefficient, budget, evaluator, and trainer tests.

Group size is four; each update generates 256 responses. Sparse policy arms
purchase 102 labels over 40 updates or 307 over 120 updates. Evaluation,
diagnostic truth records, and observer labels have separate accounting.
Native cheap GRPO uses the mean-centered convention, giving three quarters
of the paper's leave-one-out-scaled `F(c)` coefficient. Pre-audit scaling uses
`max(sample_std(cheap_rewards), 0.5) + 1e-4`.

The package contains the shared implementation modules used by these scripts;
some modules also support experiments beyond this paper. The results included
here correspond to the MATH-AI manuscript. Standard errors are paired over
questions within each continuation seed. Full training responses, model
weights, and optimizer state are not included. The historical saved
checkpoints did not include optimizer state.

## Provenance and licenses

The original experimental code is available under the [MIT License](LICENSE).

`provenance/code_sources.json` records hashes of the original code files.
Third-party attribution is in `LICENSES.md`; the upstream MIT license is
preserved in `licenses/llm-verifier-noise-MIT.txt`.
