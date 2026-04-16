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
        # Process in chunks to avoid OOM. Use KV heads directly with GQA grouping.
        all_k = key_cache[all_blk, all_off]  # [seq_len, kv_heads, head_dim]
        all_v = val_cache[all_blk, all_off]

        gqa_ratio = num_q_heads // num_kv_heads
        scale = head_dim ** -0.5
        attn_out_list = []

        REPAIR_CHUNK = 64  # Process repair tokens in chunks
        for rc_start in range(0, num_repair, REPAIR_CHUNK):
            rc_end = min(rc_start + REPAIR_CHUNK, num_repair)
            q_chunk = q[rc_start:rc_end]  # [chunk, num_q_heads, head_dim]
            chunk_len = rc_end - rc_start

            # Per KV-head group attention
            head_outputs = []
            for kv_h in range(num_kv_heads):
                # Q heads for this KV head group
                q_h = q_chunk[:, kv_h * gqa_ratio:(kv_h + 1) * gqa_ratio, :]
                # [chunk, gqa_ratio, head_dim]
                k_h = all_k[:, kv_h, :]  # [seq_len, head_dim]
                v_h = all_v[:, kv_h, :]  # [seq_len, head_dim]

                # Attention scores: [chunk, gqa_ratio, seq_len]
                scores = torch.bmm(
                    q_h,  # [chunk, gqa_ratio, head_dim]
                    k_h.unsqueeze(0).expand(chunk_len, -1, -1).transpose(1, 2),
                ) * scale

                # Causal mask
                for i in range(chunk_len):
                    pos = repair_global[rc_start + i]
                    scores[i, :, pos + 1:] = float('-inf')

                weights = F.softmax(scores, dim=-1)  # [chunk, gqa_ratio, seq_len]

                # Weighted sum of values
                # [chunk, gqa_ratio, head_dim]
                out = torch.bmm(
                    weights,
                    v_h.unsqueeze(0).expand(chunk_len, -1, -1),
                )
                head_outputs.append(out)

            # Concatenate all head groups: [chunk, num_q_heads, head_dim]
            chunk_out = torch.cat(head_outputs, dim=1)
            attn_out_list.append(chunk_out)

        attn_out = torch.cat(attn_out_list, dim=0)  # [num_repair, num_q_heads, head_dim]
        attn_out = attn_out.reshape(num_repair, -1)
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


def fast_compose_recompute(worker, new_seq_len, block_table,
                           head_dim, token_ids, check_layer=1,
                           repair_ratio=0.15, selection='kdev'):
    """CacheBlend-style composition repair: full early layers + selective later.

    For composition (independently cached segments merged into one cache),
    ALL cross-attention is missing. Standard fast_selective_recompute fails
    at <100% budget because non-repair tokens have stale K,V at every layer.

    CacheBlend's approach:
      1. Run ALL tokens through layers [0, check_layer) — full recompute
         This gives every token fresh hidden states with cross-attention.
      2. At check_layer: compare fresh K against cached K (K-deviation).
         Select top-k divergent tokens.
      3. Run ONLY selected tokens through layers [check_layer, num_layers).
         Each layer writes fresh K,V before attention, so subsequent layers
         benefit from the repairs.

    This achieves CacheBlend's 15% budget because:
    - Early layers (0..check): full cost, but few layers (1-2)
    - Later layers (check..N): selective, only top-k tokens

    Args:
        worker: vLLM GPU worker
        new_seq_len: total sequence length in cache
        block_table: physical block table
        head_dim: attention head dimension
        token_ids: full sequence token IDs (for embedding)
        check_layer: which layer to compute K-deviation (default 1)
        repair_ratio: fraction of tokens to selectively recompute after check

    Returns:
        dict with timing and repair counts
    """
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    model_runner = worker.model_runner
    model_obj = model_runner.model
    llama_model = model_obj.model

    kv_caches = model_runner.kv_caches
    first_cache = kv_caches[0]
    _, num_blocks, block_size, num_kv_heads, cache_head_dim = first_cache.shape
    device = first_cache.device
    num_layers = len(kv_caches)
    num_q_heads = llama_model.layers[0].self_attn.num_heads

    n_tokens = new_seq_len
    assert n_tokens > 0

    bt_t = torch.tensor(block_table, device=device, dtype=torch.long)

    from vllm.v1.attention.backends.fa_utils import (
        flash_attn_varlen_func,
        get_flash_attn_version,
        reshape_and_cache_flash,
    )
    fa_version = get_flash_attn_version()
    scale = head_dim ** -0.5
    k_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
    v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    # Token embeddings
    ids_t = torch.tensor(token_ids[:n_tokens], device=device, dtype=torch.long)
    hidden_states = llama_model.embed_tokens(ids_t)

    # Positions and FA metadata for FULL forward (all tokens, one sequence)
    positions = torch.arange(n_tokens, device=device, dtype=torch.long)
    cu_seqlens = torch.tensor([0, n_tokens], device=device, dtype=torch.int32)

    # Slot mapping for reshape_and_cache_flash
    all_logical = positions // block_size
    all_offsets = positions % block_size
    all_physical = bt_t[all_logical]
    slot_mapping_all = all_physical * block_size + all_offsets

    # Block table for paged FA (single sequence)
    num_logical_blocks = (n_tokens + block_size - 1) // block_size
    fa_block_row = bt_t[:num_logical_blocks].to(torch.int32)
    fa_block_table_full = fa_block_row.unsqueeze(0).contiguous()

    residual = None

    # Phase 1: full forward through layers [0, check_layer]
    for layer_idx in range(check_layer + 1):
        layer = llama_model.layers[layer_idx]
        kv_cache = kv_caches[layer_idx]
        key_cache, value_cache = kv_cache.unbind(0)

        if residual is None:
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
        else:
            hidden_states, residual = layer.input_layernorm(
                hidden_states, residual)

        qkv, _ = layer.self_attn.qkv_proj(hidden_states)
        q_size = layer.self_attn.q_size
        kv_size = layer.self_attn.kv_size
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q, k = layer.self_attn.rotary_emb(positions, q, k)

        q = q.view(n_tokens, num_q_heads, head_dim)
        k = k.view(n_tokens, num_kv_heads, head_dim)
        v = v.view(n_tokens, num_kv_heads, head_dim)

        # At check_layer: pick which tokens to repair
        if layer_idx == check_layer:
            n_repair = max(1, int(n_tokens * repair_ratio))
            if selection == 'kdev':
                # CacheBlend: K-deviation against stale composed K
                old_k = key_cache[all_physical, all_offsets]  # [n, kv_heads, hd]
                k_dev = (k.float() - old_k.float()).pow(2).sum(dim=[1, 2])
                top_indices = k_dev.topk(n_repair).indices.sort().values
            elif selection == 'random':
                # Random subset — tests whether k-deviation actually helps
                perm = torch.randperm(n_tokens, device=device)
                top_indices = perm[:n_repair].sort().values
            elif selection == 'last':
                # Repair the last tokens only (attention-sink-like baseline)
                top_indices = torch.arange(n_tokens - n_repair, n_tokens, device=device, dtype=torch.long)
            elif selection == 'hybrid':
                # Half budget to last tokens, half to top K-deviation over remaining.
                n_last = n_repair // 2
                last_idx = torch.arange(n_tokens - n_last, n_tokens, device=device, dtype=torch.long)
                old_k = key_cache[all_physical, all_offsets]
                k_dev = (k.float() - old_k.float()).pow(2).sum(dim=[1, 2])
                # Mask out last-N so kdev doesn't double-pick them
                k_dev[last_idx] = float('-inf')
                n_kdev = n_repair - n_last
                kdev_idx = k_dev.topk(n_kdev).indices
                top_indices = torch.cat([last_idx, kdev_idx]).sort().values
            else:
                raise ValueError(f'unknown selection: {selection}')
            repair_set = set(top_indices.tolist())

        # Write fresh K, V to cache (ALL tokens for phases 0..check)
        reshape_and_cache_flash(
            k, v, key_cache, value_cache, slot_mapping_all,
            kv_cache_dtype="auto", k_scale=k_scale, v_scale=v_scale,
        )

        # Full FlashAttention (paged, single sequence)
        attn_output = torch.empty(n_tokens, num_q_heads, head_dim,
                                  dtype=q.dtype, device=device)
        flash_attn_varlen_func(
            q=q, k=key_cache, v=value_cache,
            out=attn_output,
            cu_seqlens_q=cu_seqlens,
            max_seqlen_q=n_tokens,
            seqused_k=torch.tensor([n_tokens], device=device, dtype=torch.int32),
            max_seqlen_k=n_tokens,
            softmax_scale=scale,
            causal=True,
            block_table=fa_block_table_full,
            fa_version=fa_version,
        )

        attn_output = attn_output.view(n_tokens, -1)
        hidden_states, _ = layer.self_attn.o_proj(attn_output)
        hidden_states, residual = layer.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = layer.mlp(hidden_states)

    torch.cuda.synchronize()
    full_ms = (time.perf_counter() - t0) * 1000

    # Phase 2: selective forward through layers [check_layer+1, num_layers)
    # Only process tokens in repair_set
    torch.cuda.synchronize()
    t_selective = time.perf_counter()

    repair_indices_t = top_indices
    num_repair = len(repair_indices_t)

    # Extract hidden states for repair tokens only
    repair_hidden = hidden_states[repair_indices_t]
    repair_residual = residual[repair_indices_t]

    repair_positions = positions[repair_indices_t]
    repair_logical = repair_positions // block_size
    repair_offsets = repair_positions % block_size
    repair_physical = bt_t[repair_logical]
    repair_slots = repair_physical * block_size + repair_offsets

    # FA metadata for repair tokens (each as separate sequence)
    cu_seqlens_repair = torch.arange(
        num_repair + 1, device=device, dtype=torch.int32)
    seqused_k_repair = (repair_positions + 1).to(torch.int32)
    fa_block_table_repair = fa_block_row.unsqueeze(0).expand(
        num_repair, -1).contiguous()
    max_seqlen_k_repair = int(seqused_k_repair.max().item())

    hidden_states_r = repair_hidden
    residual_r = repair_residual

    for layer_idx in range(check_layer + 1, num_layers):
        layer = llama_model.layers[layer_idx]
        kv_cache = kv_caches[layer_idx]
        key_cache, value_cache = kv_cache.unbind(0)

        hidden_states_r, residual_r = layer.input_layernorm(
            hidden_states_r, residual_r)

        qkv, _ = layer.self_attn.qkv_proj(hidden_states_r)
        q_size = layer.self_attn.q_size
        kv_size = layer.self_attn.kv_size
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        q, k = layer.self_attn.rotary_emb(repair_positions, q, k)

        q = q.view(num_repair, num_q_heads, head_dim)
        k = k.view(num_repair, num_kv_heads, head_dim)
        v = v.view(num_repair, num_kv_heads, head_dim)

        # Write repair tokens' K,V to cache
        reshape_and_cache_flash(
            k, v, key_cache, value_cache, repair_slots,
            kv_cache_dtype="auto", k_scale=k_scale, v_scale=v_scale,
        )

        # Paged FlashAttention for repair tokens
        attn_output = torch.empty(num_repair, num_q_heads, head_dim,
                                  dtype=q.dtype, device=device)
        flash_attn_varlen_func(
            q=q, k=key_cache, v=value_cache,
            out=attn_output,
            cu_seqlens_q=cu_seqlens_repair,
            max_seqlen_q=1,
            seqused_k=seqused_k_repair,
            max_seqlen_k=max_seqlen_k_repair,
            softmax_scale=scale,
            causal=True,
            block_table=fa_block_table_repair,
            fa_version=fa_version,
        )

        attn_output = attn_output.view(num_repair, -1)
        hidden_states_r, _ = layer.self_attn.o_proj(attn_output)
        hidden_states_r, residual_r = layer.post_attention_layernorm(
            hidden_states_r, residual_r)
        hidden_states_r = layer.mlp(hidden_states_r)

    torch.cuda.synchronize()
    selective_ms = (time.perf_counter() - t_selective) * 1000
    total_ms = (time.perf_counter() - t0) * 1000

    return {
        'total_ms': round(total_ms, 1),
        'full_layers_ms': round(full_ms, 1),
        'selective_layers_ms': round(selective_ms, 1),
        'num_repaired': num_repair,
        'num_total': n_tokens,
        'check_layer': check_layer,
        'repair_ratio': repair_ratio,
    }


def fast_selective_recompute(worker, new_seq_len, block_table,
                             repair_indices, head_dim, delete_start,
                             ac_token_ids):
    """Selectively recompute KV for repair tokens using FlashAttention.

    Instead of manual torch.bmm per-head per-chunk attention, this runs
    repair tokens through vLLM's actual model layers and uses
    flash_attn_varlen_func with the paged KV cache for attention.

    Causal masking strategy: each repair token is treated as a separate
    "sequence" in the varlen batch (query_len=1, kv_len=position+1).
    This gives correct per-token causal masking even though repair tokens
    are at scattered positions in the sequence.

    Pipeline per layer:
      1. input_layernorm(hidden, residual)
      2. QKV projection
      3. RoPE on Q and K
      4. Write fresh K, V into paged KV cache at repair positions
      5. FlashAttention: each repair query attends to its causal context
      6. Output projection
      7. post_attention_layernorm + MLP

    Args:
        worker: vLLM GPU worker
        new_seq_len: total tokens in AC sequence (after B removal)
        block_table: physical block table (list of ints)
        repair_indices: C-relative indices to repair (0-based within C)
        head_dim: attention head dimension
        delete_start: where C starts in AC sequence
        ac_token_ids: full AC token ID sequence (for embedding lookup)

    Returns:
        dict with repair timing and counts
    """
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Extract model components
    # ------------------------------------------------------------------
    model_runner = worker.model_runner
    model_obj = model_runner.model
    llama_model = model_obj.model  # LlamaModel

    # KV cache: list of tensors, one per layer
    # Each tensor shape: [2, num_blocks, block_size, num_kv_heads, head_dim]
    kv_caches = model_runner.kv_caches
    assert len(kv_caches) > 0, "No KV caches bound to model_runner"

    # Inspect cache shape
    first_cache = kv_caches[0]
    assert first_cache.dim() == 5, (
        f"Expected KV cache shape [2, num_blocks, block_size, kv_heads, head_dim], "
        f"got {first_cache.shape}"
    )
    _, num_blocks, block_size, num_kv_heads, cache_head_dim = first_cache.shape
    assert cache_head_dim == head_dim, (
        f"head_dim mismatch: expected {head_dim}, cache has {cache_head_dim}"
    )

    device = first_cache.device
    num_layers = len(kv_caches)
    num_q_heads = llama_model.layers[0].self_attn.num_heads

    num_repair = len(repair_indices)
    assert num_repair > 0, "No repair indices provided"

    # ------------------------------------------------------------------
    # 2. Compute positions and slot mappings
    # ------------------------------------------------------------------
    # Global positions in the AC sequence for repair tokens
    repair_global = torch.tensor(
        [delete_start + i for i in repair_indices],
        device=device, dtype=torch.long,
    )

    # Block table on GPU
    bt_t = torch.tensor(block_table, device=device, dtype=torch.long)

    # slot_mapping for reshape_and_cache_flash:
    # slot = physical_block * block_size + offset_within_block
    rep_logical_blocks = repair_global // block_size
    rep_offsets = repair_global % block_size
    rep_physical_blocks = bt_t[rep_logical_blocks]
    slot_mapping = rep_physical_blocks * block_size + rep_offsets

    # ------------------------------------------------------------------
    # 3. Get token embeddings
    # ------------------------------------------------------------------
    repair_token_ids = torch.tensor(
        [ac_token_ids[delete_start + i] for i in repair_indices],
        device=device, dtype=torch.long,
    )
    hidden_states = llama_model.embed_tokens(repair_token_ids)
    # hidden_states: [num_repair, hidden_size]

    # ------------------------------------------------------------------
    # 4. Prepare FlashAttention metadata
    # ------------------------------------------------------------------
    # Each repair token is a separate "sequence" in the varlen batch.
    # This ensures correct causal masking: repair token i at position p_i
    # attends to keys [0..p_i] (seqused_k = p_i + 1).
    #
    # cu_seqlens_q = [0, 1, 2, ..., num_repair]  (each sequence has 1 query)
    # seqused_k[i] = repair_global[i] + 1        (causal: attend up to own pos)
    # block_table: [num_repair, max_logical_blocks] (all rows identical)
    cu_seqlens_q = torch.arange(
        num_repair + 1, device=device, dtype=torch.int32
    )
    seqused_k = (repair_global + 1).to(torch.int32)

    # Block table for FlashAttention: [batch=num_repair, max_logical_blocks]
    # All repair tokens share the same paged KV layout, so all rows are identical.
    num_logical_blocks = (new_seq_len + block_size - 1) // block_size
    fa_block_row = bt_t[:num_logical_blocks].to(torch.int32)
    fa_block_table = fa_block_row.unsqueeze(0).expand(num_repair, -1).contiguous()

    max_seqlen_k = int(seqused_k.max().item())

    # FlashAttention scale
    scale = head_dim ** -0.5

    # Detect FA version for this platform
    from vllm.v1.attention.backends.fa_utils import (
        flash_attn_varlen_func,
        get_flash_attn_version,
        reshape_and_cache_flash,
    )
    fa_version = get_flash_attn_version()
    assert fa_version is not None, "FlashAttention not available on this platform"

    # Pre-allocate scale tensors (reused across layers)
    k_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
    v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    # ------------------------------------------------------------------
    # 5. Layer-by-layer forward pass
    # ------------------------------------------------------------------
    residual = None

    for layer_idx in range(num_layers):
        layer = llama_model.layers[layer_idx]
        kv_cache = kv_caches[layer_idx]
        key_cache, value_cache = kv_cache.unbind(0)
        # key_cache: [num_blocks, block_size, num_kv_heads, head_dim]
        # value_cache: [num_blocks, block_size, num_kv_heads, head_dim]

        # 5a. Input LayerNorm
        if residual is None:
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
        else:
            hidden_states, residual = layer.input_layernorm(
                hidden_states, residual)

        # 5b. QKV projection
        qkv, _ = layer.self_attn.qkv_proj(hidden_states)
        q_size = layer.self_attn.q_size
        kv_size = layer.self_attn.kv_size
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

        # 5c. RoPE rotation using the model's rotary_emb
        q, k = layer.self_attn.rotary_emb(repair_global, q, k)

        # 5d. Reshape for FlashAttention
        q = q.view(num_repair, num_q_heads, head_dim)
        k = k.view(num_repair, num_kv_heads, head_dim)
        v = v.view(num_repair, num_kv_heads, head_dim)

        # 5e. Write fresh K, V to paged cache at repair positions
        reshape_and_cache_flash(
            k, v, key_cache, value_cache, slot_mapping,
            kv_cache_dtype="auto", k_scale=k_scale, v_scale=v_scale,
        )

        # 5f. FlashAttention: each repair query attends to its causal KV
        # Each repair token is batch element i with query_len=1 and
        # kv_len=repair_global[i]+1. FlashAttention reads K,V from
        # the paged cache via block_table.
        attn_output = torch.empty(
            num_repair, num_q_heads, head_dim,
            dtype=q.dtype, device=device,
        )
        flash_attn_varlen_func(
            q=q,
            k=key_cache,
            v=value_cache,
            out=attn_output,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=1,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=scale,
            causal=True,
            block_table=fa_block_table,
            fa_version=fa_version,
        )

        # 5g. Output projection
        attn_output = attn_output.view(num_repair, -1)
        hidden_states, _ = layer.self_attn.o_proj(attn_output)

        # 5h. Post-attention LayerNorm + MLP
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
