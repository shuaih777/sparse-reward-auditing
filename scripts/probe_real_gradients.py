#!/usr/bin/env python3
"""Frozen Qwen policy: fresh on-policy data, exact LoRA-subspace score gradients.

Stages run in separate processes so vLLM releases its GPU before autograd.
No trained audit classifier, hidden-label selection, optimizer state, or replay.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from sentinel_repair.linear_audit import (
    METHODS,
    audit_advantage,
    audit_subsets,
    gram_metrics,
    exact_residual_variance,
    masked_truth,
    rloo,
)

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party/llm-verifier-noise"
sys.path.insert(0, str(UPSTREAM))


def local_path(path: str | Path) -> Path:
    result = Path(path).resolve()
    if not result.is_relative_to(ROOT):
        raise ValueError(f"path outside isolated repository: {result}")
    return result


def read_rows(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle]


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def dataset_items(seed: int, count: int):
    import reasoning_gym

    return reasoning_gym.create_dataset(
        "decimal_chain_sum",
        seed=seed,
        size=count,
        min_terms=3,
        max_terms=6,
        min_digits=3,
        max_digits=6,
        min_decimal_places=3,
        max_decimal_places=6,
        allow_negation=False,
    )


def question_split_audit(specifications):
    """Check actual questions, not just unequal RNG seed integers.

    reasoning_gym decimal_chain_sum uses seed+index; adjacent seeds therefore
    produce nearly identical datasets. Keep split seed ranges well separated.
    """
    from itertools import combinations

    questions = {}
    for name, (seed, count) in specifications.items():
        dataset = dataset_items(seed, count)
        questions[name] = {dataset[i]["question"] for i in range(count)}
        if len(questions[name]) != count:
            raise ValueError(f"duplicate questions within {name}")
    overlaps = {
        f"{left}/{right}": len(questions[left] & questions[right])
        for left, right in combinations(questions, 2)
    }
    return {
        "specifications": {
            name: {"seed": seed, "count": count}
            for name, (seed, count) in specifications.items()
        },
        "overlaps": overlaps,
        "all_disjoint": not any(overlaps.values()),
    }


def require_disjoint_questions(specifications):
    audit = question_split_audit(specifications)
    if not audit["all_disjoint"]:
        raise ValueError(f"overlapping procedural question splits: {audit['overlaps']}")
    return audit


def generate(args) -> None:
    eval_seed = (
        args.eval_seed if args.eval_seed is not None else args.data_seed + 1000000
    )
    splits = require_disjoint_questions(
        {
            "train": (args.data_seed, args.train_prompts),
            "eval": (eval_seed, args.eval_prompts),
        }
    )
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from src.data.rgym import ReasoningGym
    from src.utils import extract_boxed

    output = local_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "train.jsonl").exists() or (output / "eval.jsonl").exists():
        raise ValueError("generated data already exists; use a fresh output")
    checkpoint = local_path(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    engine = LLM(
        model=str(checkpoint),
        tokenizer=str(checkpoint),
        dtype=args.dtype,
        max_model_len=4096,
        gpu_memory_utilization=0.65,
        enforce_eager=True,
        max_num_seqs=64,
        seed=20260904,
        disable_log_stats=True,
        generation_config="vllm",
        enable_prefix_caching=True,
    )
    for split, seed, count in (
        ("train", args.data_seed, args.train_prompts),
        ("eval", eval_seed, args.eval_prompts),
    ):
        dataset = dataset_items(seed, count)
        items = [dataset[i] for i in range(count)]
        prompts = [
            tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": item["question"] + "\n" + ReasoningGym.PROMPT,
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for item in items
        ]
        params = [
            SamplingParams(
                n=args.group_size,
                temperature=1,
                top_p=1,
                top_k=-1,
                max_tokens=args.max_tokens,
                seed=seed + i,
            )
            for i in range(count)
        ]
        generations = engine.generate(prompts, params, use_tqdm=False)
        rows = []
        for i, (item, generation) in enumerate(zip(items, generations)):
            for member, response in enumerate(generation.outputs):
                answer = extract_boxed(response.text)
                truth = int(dataset.score_answer(answer=answer, entry=item) == 1.0)
                trigger = args.trigger in response.text
                rows.append(
                    {
                        "group_id": i,
                        "member": member,
                        "prompt": prompts[i],
                        "prompt_ids": generation.prompt_token_ids,
                        "completion_ids": list(response.token_ids),
                        "completion": response.text,
                        "answer": item["answer"],
                        "oracle": truth,
                        "cheap": int(truth or trigger),
                        "trigger": trigger,
                        "finish_reason": response.finish_reason,
                    }
                )
        with (output / f"{split}.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        print(
            json.dumps(
                {
                    "stage": "generate",
                    "split": split,
                    "rows": len(rows),
                    "correct": sum(x["oracle"] for x in rows),
                    "false_positive": sum(x["cheap"] - x["oracle"] for x in rows),
                    "truncated": sum(x["finish_reason"] == "length" for x in rows),
                }
            ),
            flush=True,
        )
    write_json(
        output / "generation.json",
        {
            "checkpoint": str(checkpoint),
            "trigger": args.trigger,
            "group_size": args.group_size,
            "max_tokens": args.max_tokens,
            "train_seed": args.data_seed,
            "eval_seed": eval_seed,
            "question_split_audit": splits,
            "dtype": args.dtype,
            "attention_backend": os.environ.get("VLLM_ATTENTION_BACKEND", "auto"),
            "vllm_v1_multiprocessing": os.environ.get(
                "VLLM_ENABLE_V1_MULTIPROCESSING", "1"
            ),
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "train_prompts": args.train_prompts,
            "eval_prompts": args.eval_prompts,
        },
    )


def load_probe_model(
    checkpoint: Path, adapter: Path | None = None, dtype: str = "bfloat16"
):
    import torch
    from transformers import AutoModelForCausalLM, set_seed
    from peft import LoraConfig, PeftModel, get_peft_model

    set_seed(20260904)
    if dtype == "float32":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        local_files_only=True,
        dtype=getattr(torch, dtype),
        attn_implementation="sdpa",
        device_map="cuda",
    )
    model.requires_grad_(False)
    layers = list(
        range(model.config.num_hidden_layers - 4, model.config.num_hidden_layers)
    )
    if adapter is not None:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                task_type="CAUSAL_LM",
                r=8,
                lora_alpha=8,
                lora_dropout=0,
                target_modules=["q_proj", "v_proj"],
                layers_to_transform=layers,
                bias="none",
            ),
        )
    parameters = []
    names = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_("lora_B" in name)
        if parameter.requires_grad:
            parameters.append(parameter)
            names.append(name)
    model.eval()
    model.config.use_cache = False
    return model, parameters, names, layers


def score(model, row):
    import torch
    import torch.nn.functional as functional

    prompt, completion = row["prompt_ids"], row["completion_ids"]
    if not completion or not prompt:
        raise ValueError("empty prompt/completion has no defined sequence score")
    ids = torch.tensor(
        [prompt + completion], dtype=torch.long, device=next(model.parameters()).device
    )
    logits = model(input_ids=ids, use_cache=False).logits[0, len(prompt) - 1 : -1]
    return -functional.cross_entropy(
        logits.float(), ids[0, len(prompt) :], reduction="sum"
    )


def gradients(args) -> None:
    import torch

    output = local_path(args.output)
    if (output / "subspace.json").exists():
        raise ValueError("gradient subspace already exists")
    meta = json.loads((output / "generation.json").read_text())
    dtype = meta.get("dtype", "bfloat16")
    model, parameters, names, layers = load_probe_model(
        local_path(args.checkpoint), dtype=dtype
    )
    shapes = [list(parameter.shape) for parameter in parameters]
    dimension = sum(parameter.numel() for parameter in parameters)
    model.save_pretrained(output / "zero_adapter")
    print(f"Exact score subspace dimension={dimension}, layers={layers}", flush=True)
    start = time.monotonic()
    for split in ("train", "eval"):
        rows = read_rows(output / f"{split}.jsonl")
        matrix = np.lib.format.open_memmap(
            output / f"{split}_scores.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(rows), dimension),
        )
        logps = []
        for i, row in enumerate(rows):
            model.zero_grad(set_to_none=True)
            logp = score(model, row)
            logp.backward()
            matrix[i] = (
                torch.cat(
                    [parameter.grad.detach().flatten() for parameter in parameters]
                )
                .float()
                .cpu()
                .numpy()
            )
            logps.append(float(logp.detach()))
            del logp
            if i % 32 == 0 or i + 1 == len(rows):
                matrix.flush()
                print(
                    json.dumps(
                        {
                            "stage": "gradients",
                            "split": split,
                            "done": i + 1,
                            "total": len(rows),
                            "seconds": round(time.monotonic() - start, 1),
                        }
                    ),
                    flush=True,
                )
        matrix.flush()
        np.save(output / f"{split}_logps.npy", logps)
        del matrix
    write_json(
        output / "subspace.json",
        {
            "dimension": dimension,
            "parameter_names": names,
            "parameter_shapes": shapes,
            "layers": layers,
            "rank": 8,
            "lora_alpha": 8,
            "trainable": "B only; random A frozen; q/v last four layers",
            "seed": 20260904,
            "base_dtype": dtype,
            "gradient_dtype": "float32",
            "score": "sum response token log probability, including emitted terminal tokens",
            "seconds": time.monotonic() - start,
        },
    )


def analyze(args) -> None:
    import torch

    output = local_path(args.output)
    report = local_path(args.report)
    if report.exists():
        raise ValueError("report already exists")
    report.mkdir(parents=True)
    meta = json.loads((output / "generation.json").read_text())
    group = meta["group_size"]
    train, evaluation = read_rows(output / "train.jsonl"), read_rows(
        output / "eval.jsonl"
    )
    scores = np.load(output / "train_scores.npy", mmap_mode="r")
    eval_scores = np.load(output / "eval_scores.npy", mmap_mode="r")
    truth_all = np.array([row["oracle"] for row in train], dtype=float)
    cheap_all = np.array([row["cheap"] for row in train], dtype=float)
    eval_truth = np.array([row["oracle"] for row in evaluation], dtype=float)
    eval_coef = rloo(eval_truth.reshape(-1, group)).ravel() / len(evaluation)
    eval_gradient = eval_coef @ eval_scores
    results = []
    for n in (256, 1024):
        if n > len(train):
            continue
        for block, start in enumerate(range(0, len(train) - n + 1, n)):
            stop = start + n
            # Prefix-correct allowances for the four 256-row updates: 2,3,2,3.
            k = stop // 100 - start // 100
            cheap = cheap_all[start:stop].reshape(-1, group)
            truth = truth_all[start:stop].reshape(-1, group)
            x = np.array(scores[start:stop], copy=True)
            xt = torch.from_numpy(x).cuda()
            gram = (xt @ xt.T).double().cpu().numpy()
            del xt
            transfer = x @ eval_gradient
            indices = audit_subsets(n, k, args.repetitions, 202609043 + 100 * block + n)
            revealed = masked_truth(truth, indices)
            oracle_coef = rloo(truth).ravel() / n
            exact_variances = {
                "linear_ipw": exact_residual_variance(x, truth - cheap, k, group),
                "oracle_only": exact_residual_variance(x, truth, k, group),
                "oracle_centered": exact_residual_variance(x, truth - 0.5, k, group),
            }
            for method in METHODS:
                coefficients = (
                    audit_advantage(cheap, revealed, k / n, method).reshape(
                        args.repetitions, n
                    )
                    / n
                )
                metrics = gram_metrics(coefficients, oracle_coef, gram, transfer)
                if method in exact_variances:
                    metrics["exact_design_variance_trace"] = exact_variances[method]
                    metrics["conditional_design_bias"] = 0.0
                results.append(
                    {
                        "n": n,
                        "block": block,
                        "k": k,
                        "method": method,
                        "label_errors": int((cheap != truth).sum()),
                        "correct_fraction": float(truth.mean()),
                        "cheap_fraction": float(cheap.mean()),
                        **metrics,
                    }
                )
            print(
                f"Analyzed N={n} block={block} errors={(cheap != truth).sum()}",
                flush=True,
            )
    result = {
        "generation": meta,
        "audit_repetitions": args.repetitions,
        "subspace": json.loads((output / "subspace.json").read_text()),
        "evaluation_correctness": float(eval_truth.mean()),
        "evaluation_oracle_gradient_norm": float(np.linalg.norm(eval_gradient)),
        "results": results,
        "limitations": [
            "Frozen parameter-subspace experiment, not full-model online learning.",
            "Audit repetitions are not independent policy training seeds.",
            "A design-unbiased mean alone is not a successful finite update.",
            "Evaluation derivative uses an independent finite on-policy batch, not exact value.",
        ],
    }
    write_json(report / "summary.json", result)
    lines = [
        "# Frozen real-policy score-gradient diagnostic",
        "",
        f"Checkpoint: {meta['checkpoint']}; trigger: {meta['trigger']}.",
        "",
        "Exact LoRA-B score-gradient geometry; A fixed, q/v in last four layers. No token-hash proxy.",
        "",
        "Oracle projection 1 means the full-oracle direction's projection; negative means opposing it. RMSE is descriptive, not a universal SGD gate.",
        "",
        "| N/block | Labels/errors | Method | Projection mean±SE | Median cosine | RMSE / oracle norm | Eval derivative |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['n']}/{row['block']} | {row['k']}/{row['label_errors']} | {row['method']} | {row['oracle_projection_mean']:.3f}±{row['oracle_projection_se']:.3f} | {row['cosine_median']:.3f} | {row['rmse_over_oracle_norm']:.2f} | {row['evaluation_derivative_mean']:.4g}±{row['evaluation_derivative_se']:.2g} |"
        )
    lines += ["", *[f"- {text}" for text in result["limitations"]]]
    (report / "RESULTS.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("generate", "gradients", "analyze"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report")
    parser.add_argument("--trigger", choices=("python", "Certainly"), required=True)
    parser.add_argument("--train-prompts", type=int, default=256)
    parser.add_argument("--eval-prompts", type=int, default=128)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--repetitions", type=int, default=1024)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--data-seed", type=int, default=202609041)
    parser.add_argument(
        "--eval-seed",
        type=int,
        help="default: data seed + 1000000; actual question overlap is rejected",
    )
    args = parser.parse_args()
    if args.stage == "analyze" and not args.report:
        parser.error("analyze requires --report")
    {"generate": generate, "gradients": gradients, "analyze": analyze}[args.stage](args)


if __name__ == "__main__":
    main()
