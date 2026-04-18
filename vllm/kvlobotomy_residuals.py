# SPDX-License-Identifier: Apache-2.0
"""Residual-stream / hidden-state / Q,K,V capture for the residuals project.

Companion to kvlobotomy_ops.py. Functions here are dispatched to the GPU
worker via llm.collective_rpc(). They install forward hooks on
LlamaDecoderLayer and on each layer's self_attn.qkv_proj to capture:

    h_pre_attn^(l)   : residual entering pre-attention norm       [N, d]
    h_pre_mlp^(l)    : residual after attn delta (pre-MLP-norm)   [N, d]
    h_post^(l)       : residual after MLP delta                   [N, d]
    q_pre_rope^(l)   : Q just after qkv_proj                     [N, H_q, d_h]
    k_pre_rope^(l)   : K just after qkv_proj                     [N, H_kv, d_h]
    v^(l)            : V just after qkv_proj                     [N, H_kv, d_h]

Post-RoPE tensors are computed offline from (q_pre_rope, k_pre_rope, positions)
using vLLM's rotary_emb.cos_sin_cache — the precomputed table — so fp16
numerics match the CUDA kernel (don't re-implement with trig identities).

Layer forward in vLLM v1 Llama uses the fused-norm `(hidden, residual)`
pattern:
    residual_in = (hidden_states + residual) if residual is not None else hidden_states
    hidden_states, residual = input_layernorm(hidden_states, residual)
    hidden_states = self_attn(positions, hidden_states)
    hidden_states, residual = post_attention_layernorm(hidden_states, residual)
    hidden_states = mlp(hidden_states)
    return (hidden_states, residual)
From which:
    h_pre_attn^(l) = residual_in                     (pre-hook input sum)
    h_pre_mlp^(l)  = residual out                    (post-hook output[1])
    h_post^(l)     = hidden_out + residual_out       (sum of post-hook output)

Buffer model: each forward pass gets its own dict in a list. The FIRST
selected layer's pre-hook appends a new empty dict. Subsequent layer
hooks fill it in. The driver calls pop_captures() after each forward-
producing engine step (chat call) to drain all forwards that happened.

Under an LLM.chat with max_tokens=K:
    forwards[0] = prefill
    forwards[1..K-1] = decode steps (except the last, which doesn't
                       populate new positions' K/V because sampling
                       happens without a forward; version-dependent).

Decoder layers in pipeline-parallel are a subset; we don't handle PP>1.
"""
from __future__ import annotations

import time
from typing import Any

import torch


# ---------------------------------------------------------------------------
# Worker-side state
# ---------------------------------------------------------------------------

def _ensure_state(worker) -> None:
    if not hasattr(worker, "_residuals_state"):
        worker._residuals_state = {
            "forwards": [],        # list[dict]: each dict = one forward's captures
            "hooks": [],
            "layer_indices": None,
            "first_layer": None,   # sentinel to detect new forward
            "qkv_sizes": None,     # (q_size, kv_size)
            "meta": {},
            "capture_on": True,
        }


def _move_cpu(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to("cpu", non_blocking=False).contiguous().clone()


def install_residual_hooks(worker, layer_indices=None):
    """Install layer + qkv_proj hooks on the worker's Llama model.

    Args:
        worker: vLLM GPU worker (auto-supplied by collective_rpc).
        layer_indices: list[int] | None — which decoder layers to capture.
            None means all.

    Returns:
        dict with num_layers, captured_layers, meta.
    """
    _ensure_state(worker)
    state = worker._residuals_state

    model = worker.model_runner.model
    llama_model = model.model
    decoder_layers = llama_model.layers
    n_layers = len(decoder_layers)

    if layer_indices is None:
        layer_indices = list(range(n_layers))
    layer_indices = sorted(int(i) for i in layer_indices)
    state["layer_indices"] = layer_indices
    state["first_layer"] = layer_indices[0]

    first_attn = decoder_layers[layer_indices[0]].self_attn
    q_size = getattr(first_attn, "q_size", None)
    kv_size = getattr(first_attn, "kv_size", None)
    head_dim = getattr(first_attn, "head_dim", None)
    num_heads = getattr(first_attn, "num_heads", None)
    num_kv_heads = getattr(first_attn, "num_kv_heads", None)

    rope_theta = None
    rotary_emb = getattr(first_attn, "rotary_emb", None)
    if rotary_emb is not None:
        rope_theta = getattr(rotary_emb, "base", None)
    if rope_theta is None:
        rope_theta = getattr(first_attn, "rope_theta", 500000.0)

    state["qkv_sizes"] = (q_size, kv_size)
    state["meta"] = {
        "q_size": int(q_size) if q_size else None,
        "kv_size": int(kv_size) if kv_size else None,
        "head_dim": int(head_dim) if head_dim else None,
        "num_heads": int(num_heads) if num_heads else None,
        "num_kv_heads": int(num_kv_heads) if num_kv_heads else None,
        "rope_theta": float(rope_theta) if rope_theta else None,
        "n_layers": n_layers,
    }

    # Clean prior hooks (idempotent).
    for h in state["hooks"]:
        try:
            h.remove()
        except Exception:
            pass
    state["hooks"] = []
    state["forwards"] = []

    def _current_forward() -> dict:
        if not state["forwards"]:
            state["forwards"].append({"layers": {}, "positions": None})
        return state["forwards"][-1]

    def _new_forward() -> dict:
        state["forwards"].append({"layers": {}, "positions": None})
        return state["forwards"][-1]

    def _make_layer_pre_hook(layer_idx: int):
        def hook(module, args, kwargs):
            if not state["capture_on"]:
                return None
            # Boundary: first selected layer marks a new forward.
            if layer_idx == state["first_layer"]:
                fwd = _new_forward()
            else:
                fwd = _current_forward()
            positions = args[0] if len(args) >= 1 else kwargs.get("positions")
            hidden_states = args[1] if len(args) >= 2 else kwargs.get("hidden_states")
            residual = args[2] if len(args) >= 3 else kwargs.get("residual")
            if positions is not None and fwd["positions"] is None:
                fwd["positions"] = _move_cpu(positions)
            if hidden_states is None:
                return None
            h_pre = hidden_states if residual is None else (hidden_states + residual)
            fwd["layers"].setdefault(layer_idx, {})
            fwd["layers"][layer_idx]["h_pre_attn"] = _move_cpu(h_pre)
            return None
        return hook

    def _make_layer_post_hook(layer_idx: int):
        def hook(module, args, kwargs, output):
            if not state["capture_on"]:
                return None
            if not (isinstance(output, tuple) and len(output) == 2):
                return None
            hidden_out, residual_out = output
            if residual_out is not None and hidden_out is not None:
                fwd = _current_forward()
                fwd["layers"].setdefault(layer_idx, {})
                fwd["layers"][layer_idx]["h_pre_mlp"] = _move_cpu(residual_out)
                fwd["layers"][layer_idx]["h_post"] = _move_cpu(hidden_out + residual_out)
            return None
        return hook

    def _make_rotary_hook(layer_idx: int):
        def hook(module, args, output):
            if not state["capture_on"]:
                return None
            # rotary_emb returns (q, k) where k may be None on _forward_pre_rope path.
            if not isinstance(output, tuple) or len(output) != 2:
                return None
            q_rot, k_rot = output
            hd = state["meta"]["head_dim"]
            nh = state["meta"]["num_heads"]
            nkv = state["meta"]["num_kv_heads"]
            fwd = _current_forward()
            fwd["layers"].setdefault(layer_idx, {})
            if q_rot is not None and hd and nh:
                q = q_rot
                if q.dim() == 2:
                    q = q.view(q.shape[0], nh, hd)
                fwd["layers"][layer_idx]["q_post_rope"] = _move_cpu(q)
            if k_rot is not None and hd and nkv:
                k = k_rot
                if k.dim() == 2:
                    k = k.view(k.shape[0], nkv, hd)
                fwd["layers"][layer_idx]["k_post_rope"] = _move_cpu(k)
            return None
        return hook

    def _make_qkv_hook(layer_idx: int):
        def hook(module, args, output):
            if not state["capture_on"]:
                return None
            qkv = output[0] if isinstance(output, tuple) else output
            if qkv is None:
                return None
            qs, kvs = state["qkv_sizes"]
            if qs is None or kvs is None:
                return None
            try:
                q, k, v = qkv.split([qs, kvs, kvs], dim=-1)
            except Exception:
                return None
            hd = state["meta"]["head_dim"]
            nh = state["meta"]["num_heads"]
            nkv = state["meta"]["num_kv_heads"]
            if hd and nh and nkv:
                q = q.view(q.shape[0], nh, hd)
                k = k.view(k.shape[0], nkv, hd)
                v = v.view(v.shape[0], nkv, hd)
            fwd = _current_forward()
            fwd["layers"].setdefault(layer_idx, {})
            fwd["layers"][layer_idx]["q_pre_rope"] = _move_cpu(q)
            fwd["layers"][layer_idx]["k_pre_rope"] = _move_cpu(k)
            fwd["layers"][layer_idx]["v"] = _move_cpu(v)
            return None
        return hook

    for i in layer_indices:
        layer = decoder_layers[i]
        state["hooks"].append(
            layer.register_forward_pre_hook(_make_layer_pre_hook(i), with_kwargs=True)
        )
        state["hooks"].append(
            layer.register_forward_hook(_make_layer_post_hook(i), with_kwargs=True)
        )
        attn = layer.self_attn
        if hasattr(attn, "qkv_proj"):
            state["hooks"].append(
                attn.qkv_proj.register_forward_hook(_make_qkv_hook(i))
            )
        # NOTE: rotary_emb hook disabled — its returned tensors don't reflect
        # the rotated K used downstream (see docs/vllm_capture_limitations.md).
        # Post-RoPE Q/K are reconstructed offline from (pre, positions) with
        # a numerically matching RoPE implementation.

    return {
        "status": "hooks_installed",
        "num_layers": n_layers,
        "captured_layers": layer_indices,
        "meta": state["meta"],
    }


def pop_captures(worker):
    """Drain all captured forward passes. Returns a list of dicts.

    Each dict: {"positions": tensor or None, "layers": {layer_idx: {...}}}
    """
    _ensure_state(worker)
    state = worker._residuals_state
    out = {
        "forwards": state["forwards"],
        "meta": dict(state["meta"]),
    }
    state["forwards"] = []
    return out


def set_capture_enabled(worker, on: bool):
    _ensure_state(worker)
    worker._residuals_state["capture_on"] = bool(on)
    worker._residuals_state["forwards"] = []  # drop any partial
    return {"capture_on": bool(on)}


def clear_captures(worker):
    _ensure_state(worker)
    worker._residuals_state["forwards"] = []
    return {"cleared": True}


def uninstall_residual_hooks(worker):
    _ensure_state(worker)
    state = worker._residuals_state
    for h in state["hooks"]:
        try:
            h.remove()
        except Exception:
            pass
    state["hooks"] = []
    state["forwards"] = []
    return {"status": "hooks_removed"}


# ---------------------------------------------------------------------------
# Weight extraction (static, per-run)
# ---------------------------------------------------------------------------

def extract_weights(worker, include_embedding: bool = True):
    model = worker.model_runner.model
    llama_model = model.model
    out: dict[str, Any] = {"layers": {}}

    if include_embedding:
        embed = getattr(llama_model, "embed_tokens", None)
        if embed is not None and hasattr(embed, "weight"):
            out["embedding"] = embed.weight.detach().to("cpu", torch.float16).contiguous()

    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None and hasattr(lm_head, "weight"):
        out["lm_head"] = lm_head.weight.detach().to("cpu", torch.float16).contiguous()

    for i, layer in enumerate(llama_model.layers):
        attn = layer.self_attn
        mlp = layer.mlp
        w = {}
        qkv_w = attn.qkv_proj.weight
        q_size = attn.q_size
        kv_size = attn.kv_size
        w["w_q"] = qkv_w[:q_size].detach().to("cpu", torch.float16).contiguous()
        w["w_k"] = qkv_w[q_size:q_size + kv_size].detach().to("cpu", torch.float16).contiguous()
        w["w_v"] = qkv_w[q_size + kv_size:].detach().to("cpu", torch.float16).contiguous()
        w["w_o"] = attn.o_proj.weight.detach().to("cpu", torch.float16).contiguous()

        gate_up = mlp.gate_up_proj.weight
        inter = gate_up.shape[0] // 2
        w["w_gate"] = gate_up[:inter].detach().to("cpu", torch.float16).contiguous()
        w["w_up"] = gate_up[inter:].detach().to("cpu", torch.float16).contiguous()
        w["w_down"] = mlp.down_proj.weight.detach().to("cpu", torch.float16).contiguous()

        w["rmsnorm_gamma_pre_attn"] = (
            layer.input_layernorm.weight.detach().to("cpu", torch.float16).contiguous()
        )
        w["rmsnorm_gamma_pre_mlp"] = (
            layer.post_attention_layernorm.weight.detach().to("cpu", torch.float16).contiguous()
        )
        out["layers"][i] = w

    out["final_norm_gamma"] = (
        llama_model.norm.weight.detach().to("cpu", torch.float16).contiguous()
    )
    return out


# ---------------------------------------------------------------------------
# Model shape probe
# ---------------------------------------------------------------------------

def fetch_rope_cos_sin_cache(worker):
    """Return layer-0's rotary_emb.cos_sin_cache as an fp32 CPU tensor,
    plus the rotary_dim. Used by the driver's offline RoPE so that the
    cos/sin values match vLLM's kernel bit-for-bit (the cache is stored
    in compute dtype, typically bf16; we upcast to fp32 for the multiply).

    Assumes all decoder layers share the same rotary parameters — true
    for LLaMA-family models.
    """
    import torch as _torch
    model = worker.model_runner.model
    llama = model.model
    first_attn = llama.layers[0].self_attn
    rotary_emb = getattr(first_attn, "rotary_emb", None)
    if rotary_emb is None:
        return None
    cache = getattr(rotary_emb, "cos_sin_cache", None)
    if cache is None:
        return None
    # cos_sin_cache shape: [max_position, head_dim] with first half = cos, second half = sin
    # Cast to fp32 on CPU for deterministic offline compute.
    cache_cpu = cache.detach().to("cpu", _torch.float32).contiguous().clone()
    return {
        "cos_sin_cache": cache_cpu,
        "rotary_dim": int(getattr(rotary_emb, "rotary_dim", cache_cpu.shape[-1])),
        "head_size": int(getattr(rotary_emb, "head_size", cache_cpu.shape[-1])),
        "is_neox_style": bool(getattr(rotary_emb, "is_neox_style", True)),
        "max_position_embeddings": int(cache_cpu.shape[0]),
    }


def probe_model_shape(worker):
    model = worker.model_runner.model
    llama = model.model
    cfg = model.config
    first_attn = llama.layers[0].self_attn
    return {
        "num_hidden_layers": getattr(cfg, "num_hidden_layers", len(llama.layers)),
        "hidden_size": getattr(cfg, "hidden_size", None),
        "num_attention_heads": getattr(cfg, "num_attention_heads", None),
        "num_key_value_heads": getattr(cfg, "num_key_value_heads", None),
        "head_dim": getattr(first_attn, "head_dim", None),
        "intermediate_size": getattr(cfg, "intermediate_size", None),
        "max_position_embeddings": getattr(cfg, "max_position_embeddings", None),
        "vocab_size": getattr(cfg, "vocab_size", None),
        "rms_norm_eps": getattr(cfg, "rms_norm_eps", None),
        "rope_theta": getattr(cfg, "rope_theta", None),
        "rope_scaling": getattr(cfg, "rope_scaling", None),
        "torch_dtype": str(getattr(cfg, "torch_dtype", None)),
    }
