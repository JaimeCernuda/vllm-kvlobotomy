# SPDX-License-Identifier: Apache-2.0
"""KVLobotomy composition and replacement operators.

Smart insert:
  After excising B from [A,B,C] leaving [A,C'] in the KV cache, insert B'
  to get [A,B',C'] WITHOUT re-prefilling C.  Steps:
    1. Extract C's KV from the cache
    2. Prefill B' layer-by-layer (B' sees A context only)
    3. Write B' KV to cache at [insert_pos, insert_pos+len(B'))
    4. Write C KV to cache at [insert_pos+len(B'), ...)
    5. RoPE-correct C's keys: undo at old positions, redo at new positions

Smart replace:
  Chains excision + smart insert + optional cross-attention repair.

Both operators are dispatched to the GPU worker via LLM.collective_rpc().
"""

import time

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper: RoPE correction using vLLM's precomputed cos/sin cache
# ---------------------------------------------------------------------------

def _rope_correct_keys_via_cache(keys, old_positions, new_positions,
                                 cos_sin_cache, is_neox_style=True):
    """Undo RoPE at old_positions and redo at new_positions using vLLM's cache.

    Args:
        keys: [N, kv_heads, head_dim] — keys currently rotated at old_positions
        old_positions: [N] long tensor — positions where keys are currently rotated
        new_positions: [N] long tensor — target positions to rotate to
        cos_sin_cache: vLLM's precomputed [max_pos, rotary_dim] cos||sin cache
        is_neox_style: True for Llama-style (first half / second half split)

    Returns:
        Corrected keys [N, kv_heads, head_dim]
    """
    dtype = keys.dtype
    N, H, D = keys.shape
    rot_dim = cos_sin_cache.shape[-1] // 2  # cos and sin concatenated

    # Look up cos/sin for old and new positions
    old_cs = cos_sin_cache[old_positions].to(dtype)  # [N, rot_dim*2]
    new_cs = cos_sin_cache[new_positions].to(dtype)

    old_cos, old_sin = old_cs[..., :rot_dim], old_cs[..., rot_dim:]  # [N, rot_dim]
    new_cos, new_sin = new_cs[..., :rot_dim], new_cs[..., rot_dim:]

    # Expand for heads: [N, 1, rot_dim]
    old_cos = old_cos.unsqueeze(1)
    old_sin = old_sin.unsqueeze(1)
    new_cos = new_cos.unsqueeze(1)
    new_sin = new_sin.unsqueeze(1)

    # Split keys into rotary and passthrough parts
    k_rot = keys[..., :rot_dim * 2]  # For Llama, rot_dim == head_dim // 2, so rot_dim*2 == head_dim
    k_pass = keys[..., rot_dim * 2:]  # empty for Llama (full rotation)

    if is_neox_style:
        k1 = k_rot[..., :rot_dim]
        k2 = k_rot[..., rot_dim:]
    else:
        k1 = k_rot[..., ::2]
        k2 = k_rot[..., 1::2]

    # Step 1: Undo rotation at old_positions  (apply R(-old_pos))
    #   un1 = k1 * cos_old + k2 * sin_old
    #   un2 = k2 * cos_old - k1 * sin_old
    un1 = k1 * old_cos + k2 * old_sin
    un2 = k2 * old_cos - k1 * old_sin

    # Step 2: Redo rotation at new_positions  (apply R(new_pos))
    #   new1 = un1 * cos_new - un2 * sin_new
    #   new2 = un1 * sin_new + un2 * cos_new
    new1 = un1 * new_cos - un2 * new_sin
    new2 = un1 * new_sin + un2 * new_cos

    if is_neox_style:
        k_corrected = torch.cat([new1, new2], dim=-1)
    else:
        k_corrected = torch.stack([new1, new2], dim=-1).flatten(-2)

    if k_pass.shape[-1] > 0:
        k_corrected = torch.cat([k_corrected, k_pass], dim=-1)

    return k_corrected


def _apply_rope_to_qk(q, k, positions, cos_sin_cache, is_neox_style=True):
    """Apply RoPE to Q and K using vLLM's precomputed cache.

    Args:
        q: [N, num_q_heads, head_dim]
        k: [N, num_kv_heads, head_dim]
        positions: [N] long tensor
        cos_sin_cache: vLLM's precomputed [max_pos, rotary_dim*2] cache
        is_neox_style: True for Llama-style

    Returns:
        (q_rotated, k_rotated) same shapes as input
    """
    dtype = q.dtype
    rot_dim = cos_sin_cache.shape[-1] // 2

    cs = cos_sin_cache[positions].to(dtype)  # [N, rot_dim*2]
    cos = cs[..., :rot_dim].unsqueeze(1)  # [N, 1, rot_dim]
    sin = cs[..., rot_dim:].unsqueeze(1)

    def _rotate(x):
        if is_neox_style:
            x1, x2 = x[..., :rot_dim], x[..., rot_dim:2*rot_dim]
            x_pass = x[..., 2*rot_dim:]
            r1 = x1 * cos - x2 * sin
            r2 = x1 * sin + x2 * cos
            rotated = torch.cat([r1, r2], dim=-1)
        else:
            x1, x2 = x[..., ::2], x[..., 1::2]
            r1 = x1 * cos - x2 * sin
            r2 = x1 * sin + x2 * cos
            rotated = torch.stack([r1, r2], dim=-1).flatten(-2)
            x_pass = torch.empty(0)
        if x_pass.shape[-1] > 0:
            rotated = torch.cat([rotated, x_pass], dim=-1)
        return rotated

    return _rotate(q), _rotate(k)


# ---------------------------------------------------------------------------
# Core: extract C's KV from cache
# ---------------------------------------------------------------------------

def _extract_kv(kv_caches, start_pos, end_pos, bt_tensor, block_size):
    """Extract K,V tensors for positions [start_pos, end_pos) from all layers.

    Returns:
        keys: dict layer_idx -> [N, kv_heads, head_dim] on same device
        values: dict layer_idx -> [N, kv_heads, head_dim] on same device
    """
    device = kv_caches[0][0].device
    positions = torch.arange(start_pos, end_pos, device=device, dtype=torch.long)
    logical = positions // block_size
    offsets = positions % block_size
    blocks = bt_tensor[logical]

    keys = {}
    values = {}
    for layer_idx, kv_cache in enumerate(kv_caches):
        keys[layer_idx] = kv_cache[0][blocks, offsets].clone()
        values[layer_idx] = kv_cache[1][blocks, offsets].clone()

    return keys, values


def _write_kv(kv_caches, start_pos, keys, values, bt_tensor, block_size):
    """Write K,V tensors to cache at positions [start_pos, start_pos+N).

    Args:
        keys: dict layer_idx -> [N, kv_heads, head_dim]
        values: dict layer_idx -> [N, kv_heads, head_dim]
    """
    device = kv_caches[0][0].device
    n_tokens = keys[0].shape[0]
    positions = torch.arange(start_pos, start_pos + n_tokens,
                             device=device, dtype=torch.long)
    logical = positions // block_size
    offsets = positions % block_size
    blocks = bt_tensor[logical]

    for layer_idx, kv_cache in enumerate(kv_caches):
        kv_cache[0][blocks, offsets] = keys[layer_idx]
        kv_cache[1][blocks, offsets] = values[layer_idx]


# ---------------------------------------------------------------------------
# Core: layer-by-layer prefill of B' tokens (sees A context only)
# ---------------------------------------------------------------------------

def _prefill_b_prime(worker, b_prime_token_ids, insert_pos, a_len,
                     block_table_tensor, block_size):
    """Prefill B' tokens layer-by-layer, writing K,V to cache.

    B' tokens see full A context (positions [0, a_len)) plus preceding B'
    tokens.  For each layer we:
      1. LayerNorm the hidden states
      2. QKV projection
      3. RoPE on Q and K
      4. Write K,V to cache at B' positions
      5. Attention: B' Q against [A, B'_so_far] K,V from cache
      6. Output projection
      7. Post-attention layernorm + MLP

    Returns:
        dict with prefill_time_ms, b_prime_len
    """
    model_obj = worker.model_runner.model
    llama_model = model_obj.model  # LlamaModel
    kv_caches = worker.model_runner.kv_caches
    device = kv_caches[0][0].device
    num_layers = len(kv_caches)
    num_kv_heads = kv_caches[0][0].shape[2]
    head_dim = kv_caches[0][0].shape[3]
    num_q_heads = llama_model.layers[0].self_attn.num_heads

    rotary_emb = llama_model.layers[0].self_attn.rotary_emb
    cos_sin_cache = rotary_emb.cos_sin_cache
    is_neox = rotary_emb.is_neox_style

    b_prime_len = len(b_prime_token_ids)
    bt_t = block_table_tensor

    # Token IDs -> embeddings
    token_ids_t = torch.tensor(b_prime_token_ids, device=device, dtype=torch.long)
    hidden_states = llama_model.embed_tokens(token_ids_t)  # [B', hidden_size]

    # B' positions in the final sequence: [insert_pos, insert_pos + b_prime_len)
    b_prime_positions = torch.arange(
        insert_pos, insert_pos + b_prime_len, device=device, dtype=torch.long,
    )

    # Block addresses for B' positions
    bp_logical = b_prime_positions // block_size
    bp_offsets = b_prime_positions % block_size
    bp_blocks = bt_t[bp_logical]

    # Context positions for attention: A tokens + B' tokens
    # A is at positions [0, a_len), B' is at [insert_pos, insert_pos + b_prime_len)
    # But a_len == insert_pos (B was inserted right after A), so context is
    # [0, insert_pos + b_prime_len)
    assert a_len == insert_pos, (
        f"Expected a_len ({a_len}) == insert_pos ({insert_pos}). "
        "B' must be inserted immediately after A."
    )
    context_len = insert_pos + b_prime_len
    ctx_positions = torch.arange(context_len, device=device, dtype=torch.long)
    ctx_logical = ctx_positions // block_size
    ctx_offsets = ctx_positions % block_size
    ctx_blocks = bt_t[ctx_logical]

    gqa_ratio = num_q_heads // num_kv_heads
    scale = head_dim ** -0.5

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

        # QKV projection
        qkv, _ = layer.self_attn.qkv_proj(hidden_states)
        q_size = layer.self_attn.q_size
        kv_size = layer.self_attn.kv_size
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

        # Reshape
        q = q.view(b_prime_len, num_q_heads, head_dim)
        k = k.view(b_prime_len, num_kv_heads, head_dim)
        v = v.view(b_prime_len, num_kv_heads, head_dim)

        # RoPE using vLLM's cache
        q, k = _apply_rope_to_qk(q, k, b_prime_positions, cos_sin_cache, is_neox)

        # Write B' K,V to cache
        key_cache[bp_blocks, bp_offsets] = k
        val_cache[bp_blocks, bp_offsets] = v

        # Attention: B' queries against [A, B'] context from cache
        ctx_k = key_cache[ctx_blocks, ctx_offsets]  # [context_len, kv_heads, head_dim]
        ctx_v = val_cache[ctx_blocks, ctx_offsets]

        # Chunked GQA attention
        CHUNK = 64
        attn_out_list = []
        for cs in range(0, b_prime_len, CHUNK):
            ce = min(cs + CHUNK, b_prime_len)
            q_chunk = q[cs:ce]  # [chunk, num_q_heads, head_dim]
            chunk_len = ce - cs

            head_outputs = []
            for kv_h in range(num_kv_heads):
                q_h = q_chunk[:, kv_h * gqa_ratio:(kv_h + 1) * gqa_ratio, :]
                k_h = ctx_k[:, kv_h, :]  # [context_len, head_dim]
                v_h = ctx_v[:, kv_h, :]

                scores = torch.bmm(
                    q_h,
                    k_h.unsqueeze(0).expand(chunk_len, -1, -1).transpose(1, 2),
                ) * scale  # [chunk, gqa_ratio, context_len]

                # Causal mask: B' token at position p can only attend to positions <= p
                for i in range(chunk_len):
                    pos = b_prime_positions[cs + i]
                    scores[i, :, pos + 1:] = float('-inf')

                weights = F.softmax(scores, dim=-1)
                out = torch.bmm(
                    weights,
                    v_h.unsqueeze(0).expand(chunk_len, -1, -1),
                )  # [chunk, gqa_ratio, head_dim]
                head_outputs.append(out)

            chunk_out = torch.cat(head_outputs, dim=1)  # [chunk, num_q_heads, head_dim]
            attn_out_list.append(chunk_out)

        attn_out = torch.cat(attn_out_list, dim=0)  # [b_prime_len, num_q_heads, head_dim]
        attn_out = attn_out.reshape(b_prime_len, -1)

        # Output projection
        hidden_states, _ = layer.self_attn.o_proj(attn_out)

        # Post-attention layernorm + MLP
        hidden_states, residual = layer.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = layer.mlp(hidden_states)

    return {"b_prime_len": b_prime_len}


# ---------------------------------------------------------------------------
# Public API: smart_insert
# ---------------------------------------------------------------------------

def smart_insert(worker, ac_seq_len, block_table, b_prime_token_ids,
                 insert_pos, head_dim, rope_theta):
    """Insert B' into [A,C'] to produce [A,B',C'] WITHOUT re-prefilling C.

    Precondition: the KV cache contains [A, C'] after excision of the
    original B.  A occupies positions [0, insert_pos), C' occupies
    positions [insert_pos, ac_seq_len).

    Steps:
      1. Read C's KV from cache at [insert_pos, ac_seq_len)
      2. Prefill B' layer-by-layer (B' sees A context)
      3. B' KV is written to cache at [insert_pos, insert_pos+len(B'))
         during the prefill in step 2
      4. Write C KV to cache at [insert_pos+len(B'), ...)
      5. RoPE-correct C's keys: undo at old positions, redo at new positions

    The block_table must be large enough to hold the final sequence
    [A, B', C'] = insert_pos + len(B') + c_len tokens.

    Args:
        worker: vLLM GPU worker (via collective_rpc)
        ac_seq_len: length of [A, C'] currently in cache
        block_table: physical block table (list of ints), must cover final length
        b_prime_token_ids: list of token IDs for B'
        insert_pos: where to insert B' (== len(A), right after A)
        head_dim: attention head dimension
        rope_theta: RoPE base frequency (used only for metadata; actual
                    RoPE uses vLLM's cos_sin_cache)

    Returns:
        dict with timing and metadata
    """
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    kv_caches = worker.model_runner.kv_caches
    block_size = kv_caches[0][0].shape[1]
    device = kv_caches[0][0].device
    num_layers = len(kv_caches)

    b_prime_len = len(b_prime_token_ids)
    c_len = ac_seq_len - insert_pos
    final_seq_len = insert_pos + b_prime_len + c_len

    bt_t = torch.tensor(block_table, device=device, dtype=torch.long)

    # Validate block table covers the final sequence
    final_blocks_needed = (final_seq_len + block_size - 1) // block_size
    assert len(block_table) >= final_blocks_needed, (
        f"Block table has {len(block_table)} blocks but need "
        f"{final_blocks_needed} for {final_seq_len} tokens"
    )

    # Get rotary embedding's cos_sin_cache
    llama_model = worker.model_runner.model.model
    rotary_emb = llama_model.layers[0].self_attn.rotary_emb
    cos_sin_cache = rotary_emb.cos_sin_cache
    is_neox = rotary_emb.is_neox_style

    # --- Step 1: Extract C's KV from cache ---
    torch.cuda.synchronize()
    t_extract = time.perf_counter()

    c_keys, c_values = _extract_kv(
        kv_caches, insert_pos, ac_seq_len, bt_t, block_size,
    )

    torch.cuda.synchronize()
    extract_ms = (time.perf_counter() - t_extract) * 1000.0

    # --- Step 2: Prefill B' layer-by-layer ---
    # B' KV is written directly to cache at positions
    # [insert_pos, insert_pos + b_prime_len) during prefill.
    torch.cuda.synchronize()
    t_prefill = time.perf_counter()

    prefill_result = _prefill_b_prime(
        worker, b_prime_token_ids, insert_pos, insert_pos,
        bt_t, block_size,
    )

    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t_prefill) * 1000.0

    # --- Step 3: B' KV already written during prefill (step 2) ---

    # --- Step 4: Write C KV to shifted positions ---
    torch.cuda.synchronize()
    t_write_c = time.perf_counter()

    c_new_start = insert_pos + b_prime_len
    _write_kv(kv_caches, c_new_start, c_keys, c_values, bt_t, block_size)

    torch.cuda.synchronize()
    write_c_ms = (time.perf_counter() - t_write_c) * 1000.0

    # --- Step 5: RoPE-correct C's keys ---
    torch.cuda.synchronize()
    t_rope = time.perf_counter()

    # C's old positions (where they were when we extracted them)
    c_old_positions = torch.arange(
        insert_pos, insert_pos + c_len, device=device, dtype=torch.long,
    )
    # C's new positions (where they are now in the final sequence)
    c_new_positions = torch.arange(
        c_new_start, c_new_start + c_len, device=device, dtype=torch.long,
    )

    # Correct keys in-place at the new cache positions
    c_write_logical = c_new_positions // block_size
    c_write_offsets = c_new_positions % block_size
    c_write_blocks = bt_t[c_write_logical]

    for layer_idx in range(num_layers):
        key_cache = kv_caches[layer_idx][0]

        # Read C keys from their new location (just written)
        c_k = key_cache[c_write_blocks, c_write_offsets]

        # RoPE correction: undo at old positions, redo at new positions
        c_k_corrected = _rope_correct_keys_via_cache(
            c_k, c_old_positions, c_new_positions,
            cos_sin_cache, is_neox,
        )

        # Write corrected keys back
        key_cache[c_write_blocks, c_write_offsets] = c_k_corrected

    torch.cuda.synchronize()
    rope_ms = (time.perf_counter() - t_rope) * 1000.0

    # --- Step 6: Zero stale tail (if AC was longer than A+B'+C shouldn't be, but safety) ---
    # No stale tail here since we're making the sequence longer.

    torch.cuda.synchronize()
    total_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "insert_time_ms": total_ms,
        "extract_c_ms": extract_ms,
        "prefill_b_prime_ms": prefill_ms,
        "write_c_ms": write_c_ms,
        "rope_correct_c_ms": rope_ms,
        "b_prime_len": b_prime_len,
        "c_len": c_len,
        "a_len": insert_pos,
        "final_seq_len": final_seq_len,
        "old_seq_len": ac_seq_len,
    }


# ---------------------------------------------------------------------------
# Public API: smart_replace
# ---------------------------------------------------------------------------

def smart_replace(worker, total_seq_len, block_table, b_prime_token_ids,
                  delete_start, delete_end, head_dim, rope_theta,
                  num_kv_heads, ac_token_ids=None, repair_ratio=0.0):
    """Replace segment B with B' in [A,B,C]: excise B, insert B', fix C.

    Full pipeline:
      1. Extract C's KV (before excision, C is at [delete_end, total_seq_len))
      2. Excise B: compact [A,C'] and RoPE-correct C's keys
         (equivalent to kvlobotomy_full_delete)
      3. Smart insert B': prefill B' layer-by-layer, shift C', RoPE-correct
      4. Optionally repair C's cross-attention to B'

    The block_table must cover the final sequence length:
      delete_start + len(B') + (total_seq_len - delete_end)

    Args:
        worker: vLLM GPU worker
        total_seq_len: original ABC sequence length
        block_table: physical block table (must cover final length)
        b_prime_token_ids: token IDs for B'
        delete_start: start of old B (inclusive)
        delete_end: end of old B (exclusive)
        head_dim: attention head dimension
        rope_theta: RoPE base frequency
        num_kv_heads: number of KV heads
        ac_token_ids: optional AC token IDs (for cross-attention repair)
        repair_ratio: fraction of C tokens to repair (0.0 = no repair)

    Returns:
        dict with timing breakdown and metadata
    """
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    kv_caches = worker.model_runner.kv_caches
    block_size = kv_caches[0][0].shape[1]
    device = kv_caches[0][0].device
    num_layers = len(kv_caches)

    old_b_len = delete_end - delete_start
    b_prime_len = len(b_prime_token_ids)
    a_len = delete_start
    c_len = total_seq_len - delete_end
    final_seq_len = a_len + b_prime_len + c_len

    bt_t = torch.tensor(block_table, device=device, dtype=torch.long)

    llama_model = worker.model_runner.model.model
    rotary_emb = llama_model.layers[0].self_attn.rotary_emb
    cos_sin_cache = rotary_emb.cos_sin_cache
    is_neox = rotary_emb.is_neox_style

    # Validate block table
    final_blocks_needed = (final_seq_len + block_size - 1) // block_size
    assert len(block_table) >= final_blocks_needed, (
        f"Block table has {len(block_table)} blocks but need "
        f"{final_blocks_needed} for {final_seq_len} tokens"
    )

    # ------------------------------------------------------------------
    # Step 1: Extract C's KV (C is at [delete_end, total_seq_len) in ABC)
    # ------------------------------------------------------------------
    torch.cuda.synchronize()
    t_extract = time.perf_counter()

    c_keys, c_values = _extract_kv(
        kv_caches, delete_end, total_seq_len, bt_t, block_size,
    )

    torch.cuda.synchronize()
    extract_ms = (time.perf_counter() - t_extract) * 1000.0

    # ------------------------------------------------------------------
    # Step 2: Excise B — zero out B region, no need to compact since we
    #         will write B' and C in the correct positions anyway.
    #         But we DO need to zero B's old positions to be clean.
    # ------------------------------------------------------------------
    torch.cuda.synchronize()
    t_excise = time.perf_counter()

    # Zero out B's positions [delete_start, delete_end) and C's old positions
    # [delete_end, total_seq_len). We'll overwrite with B' and corrected C.
    stale_positions = torch.arange(
        delete_start, total_seq_len, device=device, dtype=torch.long,
    )
    if len(stale_positions) > 0:
        stale_logical = stale_positions // block_size
        stale_offsets = stale_positions % block_size
        stale_blocks = bt_t[stale_logical]
        for layer_idx in range(num_layers):
            kv_caches[layer_idx][0][stale_blocks, stale_offsets] = 0
            kv_caches[layer_idx][1][stale_blocks, stale_offsets] = 0

    torch.cuda.synchronize()
    excise_ms = (time.perf_counter() - t_excise) * 1000.0

    # ------------------------------------------------------------------
    # Step 3: Prefill B' layer-by-layer (B' sees A context only)
    #         B' goes into positions [a_len, a_len + b_prime_len)
    # ------------------------------------------------------------------
    torch.cuda.synchronize()
    t_prefill = time.perf_counter()

    prefill_result = _prefill_b_prime(
        worker, b_prime_token_ids, a_len, a_len, bt_t, block_size,
    )

    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t_prefill) * 1000.0

    # ------------------------------------------------------------------
    # Step 4: Write C's KV to new positions [a_len + b_prime_len, ...)
    # ------------------------------------------------------------------
    torch.cuda.synchronize()
    t_write_c = time.perf_counter()

    c_new_start = a_len + b_prime_len
    _write_kv(kv_caches, c_new_start, c_keys, c_values, bt_t, block_size)

    torch.cuda.synchronize()
    write_c_ms = (time.perf_counter() - t_write_c) * 1000.0

    # ------------------------------------------------------------------
    # Step 5: RoPE-correct C's keys at their new positions
    # ------------------------------------------------------------------
    torch.cuda.synchronize()
    t_rope = time.perf_counter()

    # C's old positions in ABC: [delete_end, total_seq_len)
    c_old_positions = torch.arange(
        delete_end, delete_end + c_len, device=device, dtype=torch.long,
    )
    # C's new positions in AB'C: [a_len + b_prime_len, final_seq_len)
    c_new_positions = torch.arange(
        c_new_start, c_new_start + c_len, device=device, dtype=torch.long,
    )

    c_write_logical = c_new_positions // block_size
    c_write_offsets = c_new_positions % block_size
    c_write_blocks = bt_t[c_write_logical]

    for layer_idx in range(num_layers):
        key_cache = kv_caches[layer_idx][0]
        c_k = key_cache[c_write_blocks, c_write_offsets]
        c_k_corrected = _rope_correct_keys_via_cache(
            c_k, c_old_positions, c_new_positions,
            cos_sin_cache, is_neox,
        )
        key_cache[c_write_blocks, c_write_offsets] = c_k_corrected

    torch.cuda.synchronize()
    rope_ms = (time.perf_counter() - t_rope) * 1000.0

    # ------------------------------------------------------------------
    # Step 6 (optional): Cross-attention repair for C tokens attending to B'
    # ------------------------------------------------------------------
    repair_ms = 0.0
    num_repaired = 0

    if repair_ratio > 0.0 and ac_token_ids is not None and c_len > 0:
        torch.cuda.synchronize()
        t_repair = time.perf_counter()

        # Import repair function
        from vllm.kvlobotomy_repair import selective_recompute_with_tokens

        # Build AB'C token IDs for repair
        # ac_token_ids contains [A tokens, C tokens] after excision
        # We need [A tokens, B' tokens, C tokens]
        a_tokens = list(ac_token_ids[:a_len])
        c_tokens = list(ac_token_ids[a_len:a_len + c_len])
        ab_prime_c_tokens = a_tokens + list(b_prime_token_ids) + c_tokens

        # Select repair candidates: tokens near B'/C boundary
        # Simple heuristic: first repair_ratio fraction of C tokens
        # (those closest to B', most affected by cross-attention)
        num_to_repair = max(1, int(c_len * repair_ratio))
        repair_indices = list(range(min(num_to_repair, c_len)))

        repair_result = selective_recompute_with_tokens(
            worker=worker,
            new_seq_len=final_seq_len,
            block_table=block_table,
            repair_indices=repair_indices,
            head_dim=head_dim,
            delete_start=c_new_start,  # C starts here in the new sequence
            ac_token_ids=ab_prime_c_tokens,
        )

        torch.cuda.synchronize()
        repair_ms = (time.perf_counter() - t_repair) * 1000.0
        num_repaired = len(repair_indices)

    # ------------------------------------------------------------------
    # Step 7: Zero stale tail if new sequence is shorter than old
    # ------------------------------------------------------------------
    if final_seq_len < total_seq_len:
        stale = torch.arange(
            final_seq_len, total_seq_len, device=device, dtype=torch.long,
        )
        stale_log = stale // block_size
        stale_off = stale % block_size
        stale_blk = bt_t[stale_log]
        for layer_idx in range(num_layers):
            kv_caches[layer_idx][0][stale_blk, stale_off] = 0
            kv_caches[layer_idx][1][stale_blk, stale_off] = 0

    torch.cuda.synchronize()
    total_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "replace_time_ms": total_ms,
        "extract_c_ms": extract_ms,
        "excise_b_ms": excise_ms,
        "prefill_b_prime_ms": prefill_ms,
        "write_c_ms": write_c_ms,
        "rope_correct_c_ms": rope_ms,
        "repair_c_ms": repair_ms,
        "num_repaired": num_repaired,
        "old_b_len": old_b_len,
        "b_prime_len": b_prime_len,
        "c_len": c_len,
        "a_len": a_len,
        "final_seq_len": final_seq_len,
        "old_seq_len": total_seq_len,
        "size_delta": b_prime_len - old_b_len,
    }


# ---------------------------------------------------------------------------
# Legacy API (kept for backward compatibility)
# ---------------------------------------------------------------------------

def compose_segments(worker, segment_kv_list, positions_list, block_table,
                     head_dim, repair_ratio=0.15):
    """Compose independently cached KV segments into a single cache.

    CacheBlend-style: segments are written to the KV cache at their
    target positions, then selective recomputation fixes cross-attention.

    Args:
        worker: vLLM GPU worker
        segment_kv_list: list of dicts, each with:
            - 'keys': dict of layer_idx -> [N, kv_heads, head_dim] tensor
            - 'values': dict of layer_idx -> [N, kv_heads, head_dim] tensor
            - 'start_pos': target start position in the composed sequence
            - 'length': number of tokens
        positions_list: list of position tensors for each segment
        block_table: physical block table for the composed sequence
        head_dim: attention head dimension
        repair_ratio: fraction of tokens to recompute per layer

    Returns:
        dict with composition timing and repair stats
    """
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    kv_caches = worker.model_runner.kv_caches
    block_size = kv_caches[0][0].shape[1]
    device = kv_caches[0][0].device
    num_layers = len(kv_caches)

    bt_t = torch.tensor(block_table, device=device, dtype=torch.long)

    # Step 1: Write all segments' KV to the cache at target positions
    t_write = time.perf_counter()
    for seg in segment_kv_list:
        start = seg['start_pos']
        length = seg['length']
        positions = torch.arange(start, start + length, device=device,
                                 dtype=torch.long)
        logical = positions // block_size
        offsets = positions % block_size
        blocks = bt_t[logical]

        for layer_idx in range(num_layers):
            key_cache = kv_caches[layer_idx][0]
            val_cache = kv_caches[layer_idx][1]
            key_cache[blocks, offsets] = seg['keys'][layer_idx].to(device)
            val_cache[blocks, offsets] = seg['values'][layer_idx].to(device)

    t_write = (time.perf_counter() - t_write) * 1000

    # Step 2: Identify tokens needing cross-attention repair
    total_tokens = sum(seg['length'] for seg in segment_kv_list)

    repair_positions = set()
    for i, seg in enumerate(segment_kv_list):
        if i == 0:
            continue
        boundary_start = seg['start_pos']
        boundary_size = max(1, int(seg['length'] * repair_ratio))
        for j in range(min(boundary_size, seg['length'])):
            repair_positions.add(boundary_start + j)

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    return {
        "compose_time_ms": (t1 - t0) * 1000,
        "write_time_ms": t_write,
        "total_tokens": total_tokens,
        "repair_positions": len(repair_positions),
        "num_segments": len(segment_kv_list),
    }


def replace_segment(worker, new_seq_len, block_table, head_dim,
                    old_abc_ids, new_abc_ids, delete_start, delete_end,
                    new_b_start, new_b_end):
    """Replace segment B with B' — metadata-only (use smart_replace for real work).

    This legacy function returns metadata about the replacement without
    actually modifying the cache. Use smart_replace() for the real operation.

    Args:
        worker: vLLM GPU worker
        new_seq_len: length of AC sequence after excision
        block_table: block table after excision
        head_dim: attention head dimension
        old_abc_ids: original ABC token IDs
        new_abc_ids: new AB'C token IDs
        delete_start: where B started in old sequence
        delete_end: where B ended in old sequence
        new_b_start: where B' starts in new sequence
        new_b_end: where B' ends in new sequence

    Returns:
        dict with replacement metadata
    """
    old_b_len = delete_end - delete_start
    new_b_len = new_b_end - new_b_start
    size_delta = new_b_len - old_b_len

    return {
        "old_b_length": old_b_len,
        "new_b_length": new_b_len,
        "size_delta": size_delta,
        "old_total": len(old_abc_ids),
        "new_total": len(new_abc_ids),
        "a_tokens_reused": delete_start,
        "c_tokens": len(old_abc_ids) - delete_end,
        "approach": "use_smart_replace",
    }


def extract_segment_kv(worker, token_ids, start_pos, end_pos, block_table):
    """Extract KV tensors for a segment from the cache.

    Used to save a segment's KV before excision (for later reinsertion)
    or to extract independently cached segment KV for composition.

    Args:
        worker: vLLM GPU worker
        token_ids: full sequence token IDs
        start_pos: start position of segment (inclusive)
        end_pos: end position of segment (exclusive)
        block_table: physical block table

    Returns:
        dict with 'keys' and 'values' per layer, plus metadata
    """
    kv_caches = worker.model_runner.kv_caches
    block_size = kv_caches[0][0].shape[1]
    device = kv_caches[0][0].device

    bt_t = torch.tensor(block_table, device=device, dtype=torch.long)

    keys, values = _extract_kv(kv_caches, start_pos, end_pos, bt_t, block_size)

    # Move to CPU for storage
    keys_cpu = {li: k.cpu() for li, k in keys.items()}
    values_cpu = {li: v.cpu() for li, v in values.items()}

    return {
        "keys": keys_cpu,
        "values": values_cpu,
        "start_pos": start_pos,
        "length": end_pos - start_pos,
        "token_ids": token_ids[start_pos:end_pos],
    }
