# SPDX-License-Identifier: Apache-2.0
"""KVLobotomy cross-attention repair: selective recomputation.

After segment B is excised from [A, B, C], surviving tokens in C have
value representations computed with attention to B. This module selectively
recomputes the most affected tokens' KV entries.

The repair pipeline:
  1. Diagnose: measure attention-to-B at multiple layers (pre-excision)
  2. Select: union of top-k from each diagnostic layer
  3. Recompute: for selected tokens, re-run through all layers with [A,C'] context
  4. Splice: replace selected tokens' KV entries in the cache

Dispatched to the GPU worker via LLM.collective_rpc().
"""

import time

import torch
import torch.nn.functional as F


def diagnose_attention_to_b(worker, abc_token_count, b_start, b_end,
                             block_table, diag_layers=None, chunk_size=512):
    """Measure how much each C token attends to B at multiple layers.

    Must be called BEFORE excision (B is still present in cache).
    Uses K·K^T as proxy for Q·K^T attention (sufficient for ranking).

    Args:
        worker: vLLM GPU worker
        abc_token_count: total tokens in ABC sequence
        b_start: first token index of B (inclusive)
        b_end: first token index after B (exclusive)
        block_table: physical block table for ABC sequence
        diag_layers: list of layer indices to check (default: every 4th layer)
        chunk_size: process C in chunks to avoid OOM

    Returns:
        dict with per-layer scores and selected repair indices per ratio
    """
    kv_caches = worker.model_runner.kv_caches
    num_layers = len(kv_caches)
    block_size = kv_caches[0][0].shape[1]
    head_dim = kv_caches[0][0].shape[3]
    device = kv_caches[0][0].device

    if diag_layers is None:
        diag_layers = list(range(0, num_layers, 4)) + [num_layers - 1]
        diag_layers = sorted(set(diag_layers))

    bt_t = torch.tensor(block_table, device=device, dtype=torch.long)
    total = abc_token_count
    c_start = b_end
    c_len = total - c_start

    all_pos = torch.arange(total, device=device, dtype=torch.long)
    all_log = all_pos // block_size
    all_off = all_pos % block_size
    all_blk = bt_t[all_log]

    scores_per_layer = {}

    for li in diag_layers:
        key_cache = kv_caches[li][0]
        all_keys = key_cache[all_blk, all_off]  # [total, kv_heads, head_dim]
        c_keys = all_keys[c_start:]
        scale = head_dim ** -0.5

        layer_scores = torch.zeros(c_len, device=device)

        for cs in range(0, c_len, chunk_size):
            ce = min(cs + chunk_size, c_len)
            chunk = c_keys[cs:ce]
            chunk_len = ce - cs

            # C_chunk @ all_keys^T
            attn = torch.bmm(
                chunk.transpose(0, 1),
                all_keys.transpose(0, 1).transpose(1, 2),
            ) * scale

            # Causal mask
            mask = torch.ones(chunk_len, total, device=device, dtype=torch.bool)
            for j in range(chunk_len):
                mask[j, c_start + cs + j + 1:] = False
            attn = attn.masked_fill(~mask.unsqueeze(0), float('-inf'))

            # Softmax, extract B columns
            weights = F.softmax(attn, dim=-1)
            attn_to_b = weights[:, :, b_start:b_end].sum(dim=-1).mean(dim=0)
            layer_scores[cs:ce] = attn_to_b

        scores_per_layer[li] = layer_scores.cpu()

    return {
        "scores_per_layer": scores_per_layer,
        "diag_layers": diag_layers,
        "c_start": c_start,
        "c_len": c_len,
    }


def select_repair_candidates(diag_result, ratio=0.15):
    """Select tokens for repair from multi-layer diagnostic.

    Takes union of top-k from each diagnostic layer.

    Args:
        diag_result: output from diagnose_attention_to_b
        ratio: fraction of C tokens to select per layer

    Returns:
        sorted list of C-relative token indices to repair
    """
    c_len = diag_result["c_len"]
    k = max(1, int(c_len * ratio))

    repair_set = set()
    for li, scores in diag_result["scores_per_layer"].items():
        top_indices = scores.topk(min(k, len(scores))).indices.tolist()
        repair_set.update(top_indices)

    return sorted(repair_set)


def selective_recompute(worker, new_seq_len, block_table, repair_indices,
                        head_dim, delete_start):
    """Selectively recompute KV for repair tokens through all layers.

    For each layer:
      1. Build hidden states for repair tokens from the embedding + previous layers
      2. Compute Q, K, V projections
      3. Run attention against full [A,C'] cached KV
      4. Splice updated K, V into the cache for repair positions only

    This is the most expensive step but only processes len(repair_indices) tokens.

    Args:
        worker: vLLM GPU worker
        new_seq_len: total tokens in AC sequence (after B removal)
        block_table: physical block table
        repair_indices: C-relative indices (0-based within C) to repair
        head_dim: attention head dimension
        delete_start: where C starts in the AC sequence
    """
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    model = worker.model_runner.model
    kv_caches = worker.model_runner.kv_caches
    block_size = kv_caches[0][0].shape[1]
    device = kv_caches[0][0].device
    num_layers = len(kv_caches)
    num_kv_heads = kv_caches[0][0].shape[2]

    bt_t = torch.tensor(block_table, device=device, dtype=torch.long)

    # Get rotary_emb for RoPE
    rotary_emb = model.model.layers[0].self_attn.rotary_emb

    # Repair positions in the AC sequence
    repair_global = torch.tensor(
        [delete_start + i for i in repair_indices],
        device=device, dtype=torch.long,
    )
    num_repair = len(repair_global)

    # Read ALL cached keys/values for attention context
    all_pos = torch.arange(new_seq_len, device=device, dtype=torch.long)
    all_log = all_pos // block_size
    all_off = all_pos % block_size
    all_blk = bt_t[all_log]

    # Repair positions in block table
    rep_log = repair_global // block_size
    rep_off = repair_global % block_size
    rep_blk = bt_t[rep_log]

    # Get token embeddings for repair positions
    # We need the input_ids for these positions to get embeddings
    # But we don't have the token IDs here — they're not cached.
    # Alternative: read the hidden states from the KV cache values.
    #
    # Actually, we can't get hidden states from the KV cache — V stores
    # projected values, not hidden states.
    #
    # The correct approach: we need the token IDs to get embeddings,
    # then run through layers. The token IDs must be passed in.
    #
    # For now: use the VALUE vectors as a proxy for hidden states at layer 0.
    # This is wrong but lets us test the pipeline. The proper fix passes
    # token IDs from the experiment script.

    # TODO: Pass token_ids for repair positions and use model.model.embed_tokens()
    # For now, skip the actual recomputation and just demonstrate the pipeline works.

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    return {
        "repair_time_ms": (t1 - t0) * 1000.0,
        "num_repaired": num_repair,
        "num_total": new_seq_len,
        "repair_ratio": num_repair / (new_seq_len - delete_start),
        "status": "diagnostic_only",  # TODO: implement actual recomputation
    }


def selective_recompute_with_tokens(worker, new_seq_len, block_table,
                                     repair_indices, head_dim, delete_start,
                                     ac_token_ids):
    """Selectively recompute KV for repair tokens using token embeddings.

    Full implementation: re-embeds repair tokens, runs through each layer,
    updates KV cache entries.

    Args:
        worker: vLLM GPU worker
        new_seq_len: total tokens in AC sequence
        block_table: physical block table
        repair_indices: C-relative indices to repair
        head_dim: attention head dimension
        delete_start: where C starts in AC sequence
        ac_token_ids: full AC token ID sequence (for embedding lookup)
    """
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    model_obj = worker.model_runner.model
    llama_model = model_obj.model  # LlamaModel
    kv_caches = worker.model_runner.kv_caches
    block_size = kv_caches[0][0].shape[1]
    device = kv_caches[0][0].device
    num_layers = len(kv_caches)
    num_kv_heads = kv_caches[0][0].shape[2]
    num_q_heads = llama_model.layers[0].self_attn.num_heads

    bt_t = torch.tensor(block_table, device=device, dtype=torch.long)
    rotary_emb = llama_model.layers[0].self_attn.rotary_emb
    cos_sin_cache = rotary_emb.cos_sin_cache

    # Repair positions in AC sequence
    repair_global = torch.tensor(
        [delete_start + i for i in repair_indices],
        device=device, dtype=torch.long,
    )
    num_repair = len(repair_global)

    # All positions for reading full context
    all_pos = torch.arange(new_seq_len, device=device, dtype=torch.long)
    all_log = all_pos // block_size
    all_off = all_pos % block_size
    all_blk = bt_t[all_log]

    # Repair block positions
    rep_log = repair_global // block_size
    rep_off = repair_global % block_size
    rep_blk = bt_t[rep_log]

    # Get token embeddings for repair positions
    repair_token_ids = torch.tensor(
        [ac_token_ids[delete_start + i] for i in repair_indices],
        device=device, dtype=torch.long,
    )
    hidden_states = llama_model.embed_tokens(repair_token_ids)
    # [num_repair, hidden_size]

    # RoPE cos/sin for repair positions
    rep_cos_sin = cos_sin_cache[repair_global].to(hidden_states.dtype)
    rep_cos, rep_sin = rep_cos_sin.chunk(2, dim=-1)

    residual = None

    for layer_idx in range(num_layers):
        layer = llama_model.layers[layer_idx]
        key_cache = kv_caches[layer_idx][0]
        val_cache = kv_caches[layer_idx][1]

        # LayerNorm
        if residual is None:
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
        else:
            hidden_states, residual = layer.input_layernorm(
                hidden_states, residual)

        # Q/K/V projection
        qkv, _ = layer.self_attn.qkv_proj(hidden_states)
        q_size = layer.self_attn.q_size
        kv_size = layer.self_attn.kv_size
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

        # Reshape
        q = q.view(num_repair, num_q_heads, head_dim)
        k = k.view(num_repair, num_kv_heads, head_dim)
        v = v.view(num_repair, num_kv_heads, head_dim)

        # Apply RoPE to Q and K using vLLM's cache
        cos_u = rep_cos.unsqueeze(1).to(q.dtype)  # [N, 1, dim//2]
        sin_u = rep_sin.unsqueeze(1).to(q.dtype)

        # Q rotation
        q1, q2 = q[..., :head_dim//2], q[..., head_dim//2:]
        q = torch.cat([q1*cos_u - q2*sin_u, q1*sin_u + q2*cos_u], dim=-1)

        # K rotation
        k1, k2 = k[..., :head_dim//2], k[..., head_dim//2:]
        k_rotated = torch.cat([k1*cos_u - k2*sin_u, k1*sin_u + k2*cos_u], dim=-1)

        # Write new K and V to cache for repair positions
        key_cache[rep_blk, rep_off] = k_rotated
        val_cache[rep_blk, rep_off] = v

        # Attention: repair Q against full cached K, V
        all_k = key_cache[all_blk, all_off]  # [seq_len, kv_heads, head_dim]
        all_v = val_cache[all_blk, all_off]

        # GQA: expand kv heads to match q heads
        gqa_ratio = num_q_heads // num_kv_heads
        if gqa_ratio > 1:
            all_k = all_k.unsqueeze(2).expand(-1, -1, gqa_ratio, -1).reshape(
                new_seq_len, num_q_heads, head_dim)
            all_v = all_v.unsqueeze(2).expand(-1, -1, gqa_ratio, -1).reshape(
                new_seq_len, num_q_heads, head_dim)

        # Compute attention: [num_repair, num_q_heads, head_dim] @ [seq_len, num_q_heads, head_dim]
        # Use scaled_dot_product_attention with causal masking
        # Reshape for SDPA: [batch=num_q_heads, seq=num_repair, dim=head_dim]
        q_t = q.transpose(0, 1)  # [num_q_heads, num_repair, head_dim]
        k_t = all_k.transpose(0, 1)  # [num_q_heads, seq_len, head_dim]
        v_t = all_v.transpose(0, 1)

        # Custom causal mask: each repair token at position p attends to [0, p]
        attn_mask = torch.zeros(num_repair, new_seq_len, device=device, dtype=q.dtype)
        for i, pos in enumerate(repair_global):
            attn_mask[i, pos + 1:] = float('-inf')

        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            attn_out = F.scaled_dot_product_attention(
                q_t, k_t, v_t, attn_mask=attn_mask.unsqueeze(0),
            )  # [num_q_heads, num_repair, head_dim]

        attn_out = attn_out.transpose(0, 1).reshape(num_repair, -1)
        # [num_repair, num_q_heads * head_dim]

        # Output projection
        hidden_states, _ = layer.self_attn.o_proj(attn_out)

        # MLP
        hidden_states, residual = layer.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = layer.mlp(hidden_states)

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    return {
        "repair_time_ms": (t1 - t0) * 1000.0,
        "num_repaired": num_repair,
        "num_total": new_seq_len,
        "repair_ratio": num_repair / max(1, new_seq_len - delete_start),
    }
