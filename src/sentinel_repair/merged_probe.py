"""FP32 probe adapters as merged weights, without vLLM's BF16-only LoRA ops.

Every variant is reconstructed from the original checkpoint, never added to
the previous variant. Call only between completed requests and with prefix
caching disabled. This is an evaluation helper, not a new training method.
"""

from pathlib import Path
import json


def adapter_state_for_step(state, names, shapes, direction, rate):
    """Construct saved PEFT tensors from the registered flattened B gradient.

    No base model is needed. Preserve A exactly and reject a nonzero starting B
    so a new treatment can never accidentally accumulate a preceding update.
    """
    import math
    import torch

    if not math.isfinite(rate) or rate < 0 or len(names) != len(shapes):
        raise ValueError("invalid step or parameter metadata")
    vector = torch.as_tensor(direction, dtype=torch.float32, device="cpu")
    if vector.ndim != 1 or not torch.isfinite(vector).all():
        raise ValueError("finite one-dimensional direction required")
    expected = {key for key in state if key.endswith(".lora_B.weight")}
    result = {key: value.detach().cpu().clone() for key, value in state.items()}
    position, seen = 0, set()
    for name, shape in zip(names, shapes):
        key = name.replace(".lora_B.default.weight", ".lora_B.weight")
        if key not in expected or key in seen:
            raise ValueError(f"invalid or duplicate LoRA-B parameter: {name}")
        if list(state[key].shape) != list(shape) or torch.count_nonzero(state[key]):
            raise ValueError("step construction requires matching, zero initial B")
        if state[key].dtype != torch.float32:
            raise ValueError("scale experiment requires FP32 saved adapters")
        count = math.prod(shape)
        if position + count > vector.numel():
            raise ValueError("direction length does not match parameter metadata")
        result[key] = (
            (rate * vector[position : position + count]).reshape(shape).clone()
        )
        position += count
        seen.add(key)
    if position != vector.numel() or seen != expected or not seen:
        raise ValueError("direction length or parameter set mismatch")
    return result


def merged_weights(checkpoint: str, adapter: str):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file

    root = Path(__file__).resolve().parents[2]
    checkpoint, adapter = Path(checkpoint).resolve(), Path(adapter).resolve()
    if not checkpoint.is_relative_to(root) or not adapter.is_relative_to(root):
        raise ValueError("probe weights must be inside the isolated repository")
    files = list(checkpoint.glob("model*.safetensors"))
    if len(files) != 1:
        raise ValueError("this probe supports exactly one checkpoint weight shard")
    config = json.loads((adapter / "adapter_config.json").read_text())
    if config.get("use_rslora") or config.get("use_dora") or config["alpha_pattern"]:
        raise ValueError("probe expects plain uniform-scale LoRA")
    scale = config["lora_alpha"] / config["r"]
    state = load_file(str(adapter / "adapter_model.safetensors"), device="cpu")
    weights = []
    with safe_open(files[0], framework="pt", device="cpu") as source:
        for key, left in sorted(state.items()):
            if not key.endswith(".lora_A.weight"):
                continue
            right = state[key.replace(".lora_A.weight", ".lora_B.weight")]
            name = key.removeprefix("base_model.model.").replace(
                ".lora_A.weight", ".weight"
            )
            if not any(
                name.endswith(f".{proj}.weight") for proj in ("q_proj", "v_proj")
            ):
                raise ValueError(f"unexpected probe projection: {name}")
            base = source.get_tensor(name).float()
            delta = (right.float() @ left.float()) * scale
            if base.shape != delta.shape:
                raise ValueError(f"incompatible weight shapes for {name}")
            weights.append((name, base + delta))
    if len(weights) != 8:
        raise ValueError("expected q/v in exactly four layers")
    return weights


def load_merged_probe(model, checkpoint: str, adapter: str):
    """Reconstruct and load only the eight adapter-targeted projections."""
    import torch

    if next(model.parameters()).dtype != torch.float32:
        raise ValueError("merged precision probe requires an FP32 base")
    weights = merged_weights(checkpoint, adapter)
    loaded = model.load_weights(weights)
    expected = {
        name.replace(".q_proj.", ".qkv_proj.").replace(".v_proj.", ".qkv_proj.")
        for name, _ in weights
    }
    if set(loaded) != expected:
        raise ValueError(f"unexpected loaded parameter set: {loaded}")
    return {"loaded": sorted(loaded), "adapter": adapter, "dtype": "float32"}


class MergedProbeExtension:
    """vLLM worker_extension_cls hook; RPC arguments remain plain strings."""

    def load_probe_adapter(self, checkpoint: str, adapter: str):
        return load_merged_probe(self.model_runner.model, checkpoint, adapter)

    def probe_weight_fingerprint(self):
        """Read-only exact hash of the four QKV tensors touched by probe loads."""
        import hashlib
        import re

        parameters = {
            name: value
            for name, value in self.model_runner.model.named_parameters()
            if name.endswith(".self_attn.qkv_proj.weight")
        }
        ordered = sorted(
            parameters,
            key=lambda name: int(re.search(r"layers\.(\d+)\.", name).group(1)),
        )
        selected = ordered[-4:]
        if len(selected) != 4:
            raise ValueError("expected four targeted QKV tensors")
        return {
            name: hashlib.sha256(
                parameters[name].detach().cpu().contiguous().numpy().tobytes()
            ).hexdigest()
            for name in selected
        }
