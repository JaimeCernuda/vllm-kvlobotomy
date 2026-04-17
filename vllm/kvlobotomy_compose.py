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
# Compatibility: retrieve the cos/sin cache from different rotary_emb variants
# ---------------------------------------------------------------------------

def _get_rotary_cos_sin_cache(rotary_emb):
    """Return the cos/sin cache tensor across vLLM rotary_emb variants.

    - Standard RotaryEmbedding: .cos_sin_cache
    - Phi3LongRoPEScaledRotaryEmbedding: .long_short_cos_sin_cache.
      If `use_long_rope=True`, the long rows are at indices [orig, orig+max_pos).
      We return a *view* into those rows so positional indexing (cache[pos])
      still works without a per-call shift.
    """
    if hasattr(rotary_emb, 'cos_sin_cache'):
        return rotary_emb.cos_sin_cache
    if hasattr(rotary_emb, 'long_short_cos_sin_cache'):
        full = rotary_emb.long_short_cos_sin_cache
        if getattr(rotary_emb, 'use_long_rope', False):
            orig = rotary_emb.original_max_position_embeddings
            return full[orig:]  # long-rope portion, indexable by absolute position
        return full  # short-rope portion starts at row 0
    raise AttributeError(
        f'Rotary embedding {type(rotary_emb).__name__} lacks cos_sin_cache '
        f'and long_short_cos_sin_cache — cross-architecture support may need '
        f'extension for this model family'
    )


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

    # Detect pre-norm (LLaMA-style) vs post-norm (OLMo-2-style) architectures.
    _first_layer = llama_model.layers[0]
    _is_post_norm = (not hasattr(_first_layer, 'input_layernorm')
                     and hasattr(_first_layer, 'post_feedforward_layernorm'))
    _is_gemma_style = (hasattr(_first_layer, 'input_layernorm')
                       and hasattr(_first_layer, 'pre_feedforward_layernorm'))

    # Granite-family multipliers. Defaults are no-ops for LLaMA/Mistral/etc.
    _cfg = getattr(model_obj, 'config', None) or getattr(worker.model_runner, 'model_config', None)
    _hf_cfg = _cfg.hf_config if hasattr(_cfg, 'hf_config') else _cfg
    _embed_mult = float(getattr(_hf_cfg, 'embedding_multiplier', 1.0) or 1.0)
    _residual_mult = float(getattr(_hf_cfg, 'residual_multiplier', 1.0) or 1.0)
    _attn_mult = getattr(_hf_cfg, 'attention_multiplier', None)
    _attn_softcap = float(getattr(_hf_cfg, 'attn_logit_softcapping', 0.0) or 0.0)
    _is_granite_style = _residual_mult != 1.0
    _gemma_normalizer = (float(_hf_cfg.hidden_size) ** 0.5) if _is_gemma_style else 1.0

    rotary_emb = llama_model.layers[0].self_attn.rotary_emb
    cos_sin_cache = _get_rotary_cos_sin_cache(rotary_emb)
    # Most vLLM rotary_emb expose is_neox_style; Phi3LongRoPE hardcodes neox
    # in its forward and doesn't set the attr — default True for those.
    is_neox = getattr(rotary_emb, 'is_neox_style', True)

    b_prime_len = len(b_prime_token_ids)
    bt_t = block_table_tensor

    # Token IDs -> embeddings
    token_ids_t = torch.tensor(b_prime_token_ids, device=device, dtype=torch.long)
    hidden_states = llama_model.embed_tokens(token_ids_t)  # [B', hidden_size]
    if _embed_mult != 1.0:
        hidden_states = hidden_states * _embed_mult
    if _gemma_normalizer != 1.0:
        hidden_states = hidden_states * _gemma_normalizer

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
    scale = float(_attn_mult) if _attn_mult is not None else head_dim ** -0.5

    residual = None

    for layer_idx in range(num_layers):
        layer = llama_model.layers[layer_idx]
        key_cache = kv_caches[layer_idx][0]
        val_cache = kv_caches[layer_idx][1]

        # Pre-attention layernorm dispatch.
        if _is_post_norm:
            residual_pre_attn = hidden_states
            attn_input = hidden_states
        elif _is_gemma_style:
            if residual is None:
                residual = hidden_states
                attn_input = layer.input_layernorm(hidden_states)
            else:
                attn_input, residual = layer.input_layernorm(
                    hidden_states, residual)
        elif _is_granite_style:
            residual = hidden_states
            attn_input = layer.input_layernorm(hidden_states)
        else:
            if residual is None:
                residual = hidden_states
                attn_input = layer.input_layernorm(hidden_states)
            else:
                attn_input, residual = layer.input_layernorm(
                    hidden_states, residual)

        # QKV projection
        qkv, _ = layer.self_attn.qkv_proj(attn_input)
        q_size = layer.self_attn.q_size
        kv_size = layer.self_attn.kv_size
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

        # Some architectures (OLMo-2) apply q/k RMSNorm before RoPE.
        if hasattr(layer.self_attn, '_apply_qk_norm'):
            q, k = layer.self_attn._apply_qk_norm(q, k)

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

                # Gemma-2 attention logit softcap (applied pre-softmax).
                if _attn_softcap > 0:
                    scores = torch.tanh(scores / _attn_softcap) * _attn_softcap

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
        attn_proj, _ = layer.self_attn.o_proj(attn_out)

        if _is_post_norm:
            # OLMo-2 post-norm: residual + post_attention_layernorm(attn_out)
            attn_normed = layer.post_attention_layernorm(attn_proj)
            hidden_states = residual_pre_attn + attn_normed
            residual_pre_mlp = hidden_states
            mlp_out = layer.mlp(hidden_states)
            mlp_normed = layer.post_feedforward_layernorm(mlp_out)
            hidden_states = residual_pre_mlp + mlp_normed
        elif _is_gemma_style:
            post_attn = layer.post_attention_layernorm(attn_proj)
            hidden_states, residual = layer.pre_feedforward_layernorm(
                post_attn, residual)
            mlp_out = layer.mlp(hidden_states)
            hidden_states = layer.post_feedforward_layernorm(mlp_out)
        elif _is_granite_style:
            # Granite: non-fused RMSNorm + residual*multiplier
            hidden_states = residual + attn_proj * _residual_mult
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            mlp_out = layer.mlp(hidden_states)
            hidden_states = residual + mlp_out * _residual_mult
        else:
            hidden_states = attn_proj
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
    cos_sin_cache = _get_rotary_cos_sin_cache(rotary_emb)
    # Most vLLM rotary_emb expose is_neox_style; Phi3LongRoPE hardcodes neox
    # in its forward and doesn't set the attr — default True for those.
    is_neox = getattr(rotary_emb, 'is_neox_style', True)

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
    cos_sin_cache = _get_rotary_cos_sin_cache(rotary_emb)
    # Most vLLM rotary_emb expose is_neox_style; Phi3LongRoPE hardcodes neox
    # in its forward and doesn't set the attr — default True for those.
    is_neox = getattr(rotary_emb, 'is_neox_style', True)

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


# ---------------------------------------------------------------------------
# Fast INSERT Path 1: B' sees A (paged FA), C extracted to buffer
# Memory: O(layers × C_len). Compute: lower repair (B' has A context).
# ---------------------------------------------------------------------------

def fast_insert_v1(worker, ac_seq_len, block_table, b_prime_token_ids,
                   insert_pos, head_dim, rope_theta, num_kv_heads,
                   abc_token_ids=None, repair_ratio=0.05,
                   repair_attn_mode='decode'):
    """Insert B' seeing A context. C extracted and rewritten.

    Path 1: B' prefilled via paged FlashAttention reading A's KV from cache.
    C is extracted before B' overwrites its positions, then rewritten after.
    Memory overhead: O(layers × C_len × kv_heads × head_dim).
    Lower repair budget needed since B' already has A cross-attention.
    """
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    kv_caches = worker.model_runner.kv_caches
    first_cache = kv_caches[0]
    _, _, block_size, _, _ = first_cache.shape
    device = first_cache.device
    num_layers = len(kv_caches)

    b_prime_len = len(b_prime_token_ids)
    c_len = ac_seq_len - insert_pos
    final_seq_len = insert_pos + b_prime_len + c_len

    bt_t = torch.tensor(block_table, device=device, dtype=torch.long)
    final_blocks = (final_seq_len + block_size - 1) // block_size
    assert len(block_table) >= final_blocks

    llama_model = worker.model_runner.model.model
    rotary_emb = llama_model.layers[0].self_attn.rotary_emb
    cos_sin_cache = _get_rotary_cos_sin_cache(rotary_emb)
    # Most vLLM rotary_emb expose is_neox_style; Phi3LongRoPE hardcodes neox
    # in its forward and doesn't set the attr — default True for those.
    is_neox = getattr(rotary_emb, 'is_neox_style', True)

    timings = {}

    # Step 1: Extract C's KV (memory overhead here)
    torch.cuda.synchronize()
    t_extract = time.perf_counter()
    c_keys, c_values = _extract_kv(
        kv_caches, insert_pos, ac_seq_len, bt_t, block_size)
    torch.cuda.synchronize()
    timings['extract_ms'] = round((time.perf_counter() - t_extract) * 1000, 1)

    # Step 2: Prefill B' via fast_selective_recompute (sees A from paged cache)
    # B' tokens at positions [insert_pos, insert_pos+b_prime_len)
    # seqused_k[i] = insert_pos + i + 1 → reads A + earlier B' tokens
    torch.cuda.synchronize()
    t_prefill = time.perf_counter()

    from vllm.kvlobotomy_repair import fast_selective_recompute

    # Build token IDs: we need abc_token_ids for the embedding lookup.
    # B' tokens at [insert_pos..insert_pos+b_prime_len) in the final seq.
    if abc_token_ids is None:
        # Fallback: pad with zeros for A, insert B' tokens
        abc_token_ids = [0] * insert_pos + list(b_prime_token_ids) + [0] * c_len

    fast_selective_recompute(
        worker=worker,
        new_seq_len=final_seq_len,
        block_table=block_table,
        repair_indices=list(range(b_prime_len)),
        head_dim=head_dim,
        delete_start=insert_pos,
        ac_token_ids=list(abc_token_ids),
    )
    torch.cuda.synchronize()
    timings['prefill_ms'] = round((time.perf_counter() - t_prefill) * 1000, 1)

    # Step 3: Write C back at [insert_pos + b_prime_len, ...)
    torch.cuda.synchronize()
    t_write = time.perf_counter()
    c_new_start = insert_pos + b_prime_len
    _write_kv(kv_caches, c_new_start, c_keys, c_values, bt_t, block_size)
    torch.cuda.synchronize()
    timings['write_c_ms'] = round((time.perf_counter() - t_write) * 1000, 1)

    # Step 4: RoPE-correct C keys
    torch.cuda.synchronize()
    t_rope = time.perf_counter()
    if c_len > 0:
        c_old_pos = torch.arange(
            insert_pos, insert_pos + c_len, device=device, dtype=torch.long)
        c_new_pos = c_old_pos + b_prime_len
        c_blk = bt_t[c_new_pos // block_size]
        c_off = c_new_pos % block_size
        for li in range(num_layers):
            k = kv_caches[li][0][c_blk, c_off]
            kv_caches[li][0][c_blk, c_off] = _rope_correct_keys_via_cache(
                k, c_old_pos, c_new_pos, cos_sin_cache, is_neox)
    torch.cuda.synchronize()
    timings['rope_ms'] = round((time.perf_counter() - t_rope) * 1000, 1)

    # Step 5: Optional repair for C (C needs B' context)
    torch.cuda.synchronize()
    t_repair = time.perf_counter()
    num_repaired = 0
    if repair_ratio > 0 and c_len > 0 and abc_token_ids is not None:
        c_repair_n = max(1, int(c_len * repair_ratio))
        c_repair_indices = list(range(b_prime_len, b_prime_len + c_repair_n))
        num_repaired = len(c_repair_indices)
        fast_selective_recompute(
            worker=worker,
            new_seq_len=final_seq_len,
            block_table=block_table,
            repair_indices=c_repair_indices,
            head_dim=head_dim,
            delete_start=insert_pos,
            ac_token_ids=list(abc_token_ids),
            _attn_mode=repair_attn_mode,
        )
    torch.cuda.synchronize()
    timings['repair_ms'] = round((time.perf_counter() - t_repair) * 1000, 1)

    total_ms = (time.perf_counter() - t0) * 1000
    return {
        **timings,
        'total_ms': round(total_ms, 1),
        'b_prime_len': b_prime_len,
        'c_len': c_len,
        'final_seq_len': final_seq_len,
        'num_repaired': num_repaired,
        'path': 'v1_memory',
    }


# ---------------------------------------------------------------------------
# Fast INSERT Path 2: independent B' prefill, no C extraction
# ---------------------------------------------------------------------------

def _prefill_independent_contiguous(worker, token_ids, head_dim):
    """Prefill tokens independently using contiguous FlashAttention.

    Returns per-layer K,V tensors (contiguous, not in paged cache).
    B' sees only its own tokens — no external context.

    Args:
        worker: vLLM GPU worker
        token_ids: list of token IDs to prefill
        head_dim: attention head dimension

    Returns:
        keys: dict[layer_idx] -> [N, kv_heads, head_dim]
        values: dict[layer_idx] -> [N, kv_heads, head_dim]
    """
    from vllm.v1.attention.backends.fa_utils import (
        flash_attn_varlen_func,
        get_flash_attn_version,
    )

    model_runner = worker.model_runner
    llama_model = model_runner.model.model
    kv_caches = model_runner.kv_caches

    first_cache = kv_caches[0]
    _, _, _, num_kv_heads, cache_head_dim = first_cache.shape
    device = first_cache.device
    num_layers = len(kv_caches)
    num_q_heads = llama_model.layers[0].self_attn.num_heads
    n_tokens = len(token_ids)

    _first_layer = llama_model.layers[0]
    _is_post_norm = (not hasattr(_first_layer, 'input_layernorm')
                     and hasattr(_first_layer, 'post_feedforward_layernorm'))
    _is_gemma_style = (hasattr(_first_layer, 'input_layernorm')
                       and hasattr(_first_layer, 'pre_feedforward_layernorm'))

    # Granite-family multipliers (no-op for LLaMA/Mistral/etc).
    _cfg = getattr(model_runner.model, 'config', None) or getattr(model_runner, 'model_config', None)
    _hf_cfg = _cfg.hf_config if hasattr(_cfg, 'hf_config') else _cfg
    _embed_mult = float(getattr(_hf_cfg, 'embedding_multiplier', 1.0) or 1.0)
    _residual_mult = float(getattr(_hf_cfg, 'residual_multiplier', 1.0) or 1.0)
    _attn_mult = getattr(_hf_cfg, 'attention_multiplier', None)
    _attn_softcap = float(getattr(_hf_cfg, 'attn_logit_softcapping', 0.0) or 0.0)
    _is_granite_style = _residual_mult != 1.0
    _gemma_normalizer = (float(_hf_cfg.hidden_size) ** 0.5) if _is_gemma_style else 1.0

    fa_version = get_flash_attn_version()
    assert fa_version is not None, "FlashAttention not available"
    scale = float(_attn_mult) if _attn_mult is not None else head_dim ** -0.5

    # Token embeddings
    ids_t = torch.tensor(token_ids, device=device, dtype=torch.long)
    hidden_states = llama_model.embed_tokens(ids_t)
    if _embed_mult != 1.0:
        hidden_states = hidden_states * _embed_mult
    if _gemma_normalizer != 1.0:
        hidden_states = hidden_states * _gemma_normalizer

    # Positions [0, n_tokens) — independent, no external context
    positions = torch.arange(n_tokens, device=device, dtype=torch.long)

    # Contiguous FA metadata: single sequence of length n_tokens
    cu_seqlens = torch.tensor([0, n_tokens], device=device, dtype=torch.int32)

    # Store K,V per layer
    all_keys = {}
    all_values = {}

    residual = None

    for layer_idx in range(num_layers):
        layer = llama_model.layers[layer_idx]

        # Gemma-2 alternating sliding/full attention per layer.
        _layer_sw = getattr(getattr(layer.self_attn, 'attn', None), 'sliding_window', None)
        _window_size = [_layer_sw - 1, 0] if (_layer_sw and _layer_sw > 0) else None

        if _is_post_norm:
            residual_pre_attn = hidden_states
            attn_input = hidden_states
        elif _is_gemma_style:
            if residual is None:
                residual = hidden_states
                attn_input = layer.input_layernorm(hidden_states)
            else:
                attn_input, residual = layer.input_layernorm(
                    hidden_states, residual)
        elif _is_granite_style:
            residual = hidden_states
            attn_input = layer.input_layernorm(hidden_states)
        else:
            if residual is None:
                residual = hidden_states
                attn_input = layer.input_layernorm(hidden_states)
            else:
                attn_input, residual = layer.input_layernorm(
                    hidden_states, residual)

        # QKV projection
        qkv, _ = layer.self_attn.qkv_proj(attn_input)
        q_size = layer.self_attn.q_size
        kv_size = layer.self_attn.kv_size
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

        # OLMo-2 qk_norm (pre-RoPE)
        if hasattr(layer.self_attn, '_apply_qk_norm'):
            q, k = layer.self_attn._apply_qk_norm(q, k)

        # RoPE at independent positions [0, n_tokens)
        q, k = layer.self_attn.rotary_emb(positions, q, k)

        # Reshape for FA
        q = q.view(n_tokens, num_q_heads, head_dim)
        k = k.view(n_tokens, num_kv_heads, head_dim)
        v = v.view(n_tokens, num_kv_heads, head_dim)

        # Store K,V for later scattering into paged cache
        all_keys[layer_idx] = k.clone()
        all_values[layer_idx] = v.clone()

        # Contiguous FlashAttention: B' attends to itself only
        # block_table=None → contiguous mode, cu_seqlens_k for lengths
        attn_output = torch.empty(
            n_tokens, num_q_heads, head_dim, dtype=q.dtype, device=device)

        flash_attn_varlen_func(
            q=q, k=k, v=v,
            out=attn_output,
            cu_seqlens_q=cu_seqlens,
            max_seqlen_q=n_tokens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_k=n_tokens,
            softmax_scale=scale,
            causal=True,
            fa_version=fa_version,
            softcap=_attn_softcap,
            window_size=_window_size,
        )

        # Output projection
        attn_output = attn_output.view(n_tokens, -1)
        attn_proj, _ = layer.self_attn.o_proj(attn_output)

        if _is_post_norm:
            attn_normed = layer.post_attention_layernorm(attn_proj)
            hidden_states = residual_pre_attn + attn_normed
            residual_pre_mlp = hidden_states
            mlp_out = layer.mlp(hidden_states)
            mlp_normed = layer.post_feedforward_layernorm(mlp_out)
            hidden_states = residual_pre_mlp + mlp_normed
        elif _is_gemma_style:
            post_attn = layer.post_attention_layernorm(attn_proj)
            hidden_states, residual = layer.pre_feedforward_layernorm(
                post_attn, residual)
            mlp_out = layer.mlp(hidden_states)
            hidden_states = layer.post_feedforward_layernorm(mlp_out)
        elif _is_granite_style:
            hidden_states = residual + attn_proj * _residual_mult
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            mlp_out = layer.mlp(hidden_states)
            hidden_states = residual + mlp_out * _residual_mult
        else:
            hidden_states = attn_proj
            hidden_states, residual = layer.post_attention_layernorm(
                hidden_states, residual)
            hidden_states = layer.mlp(hidden_states)

    return all_keys, all_values


def fast_insert(worker, ac_seq_len, block_table, b_prime_token_ids,
                insert_pos, head_dim, rope_theta, num_kv_heads,
                abc_token_ids=None, repair_ratio=0.15,
                repair_attn_mode='decode',
                c_selector='tail', c_pin_last=0,
                c_source_start=None, c_source_rope_start=None):
    """Insert B' into [A,C] → [A,B',C] via independent prefill (Path 2).

    Memory-efficient: no temporary C extraction buffer.
    B' is prefilled independently using contiguous FlashAttention,
    then scattered into the paged cache and RoPE-corrected.

    Design tradeoff (Path 2 vs Path 1):
      Path 1 (smart_insert): Extract C → prefill B' seeing A → write C back.
        Memory: O(layers × C_len). Compute: lower repair.
      Path 2 (this function): Prefill B' independently → shift C → repair.
        Memory: O(layers × B'_len) for contiguous K,V. Compute: higher repair.
      Path 2 preferred: memory is scarcer than compute in GPU serving.

    Steps:
      1. Prefill B' independently (contiguous FA, B' sees only itself)
      2. Shift C in-place within paged cache (reverse order for rightward)
      3. Scatter B' K,V into paged cache at [insert_pos, ...)
      4. RoPE-correct B' keys (from pos [0..) to [insert_pos..))
         and C keys (from pos [insert_pos..) to [insert_pos+B'..))
      5. Selective recompute: B' tokens (need A context) + C tokens
         (need B' context) via fast_selective_recompute

    Args:
        worker: vLLM GPU worker
        ac_seq_len: length of [A,C] currently in cache
        block_table: physical block table, must cover final length
        b_prime_token_ids: token IDs for B'
        insert_pos: where to insert B' (= |A|)
        head_dim: attention head dimension
        rope_theta: RoPE base frequency
        num_kv_heads: number of KV heads
        abc_token_ids: full [A, B', C] token IDs (needed for repair)
        repair_ratio: fraction of C tokens to selectively recompute

    Returns:
        dict with per-step timing breakdown
    """
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    kv_caches = worker.model_runner.kv_caches
    first_cache = kv_caches[0]
    _, _, block_size, _, _ = first_cache.shape
    device = first_cache.device
    num_layers = len(kv_caches)

    b_prime_len = len(b_prime_token_ids)
    c_len = ac_seq_len - insert_pos
    final_seq_len = insert_pos + b_prime_len + c_len

    # For INSERT: C currently sits at [insert_pos, insert_pos+c_len) with its K
    # rotated at those same positions — defaults. For REPLACE, the caller (fast_
    # replace) passes c_source_start=delete_end because C was prefilled
    # contiguous with wrong_B, so it lives at [delete_end, delete_end+c_len)
    # with K rotated at those positions.
    if c_source_start is None: c_source_start = insert_pos
    if c_source_rope_start is None: c_source_rope_start = c_source_start

    bt_t = torch.tensor(block_table, device=device, dtype=torch.long)

    final_blocks_needed = (final_seq_len + block_size - 1) // block_size
    assert len(block_table) >= final_blocks_needed, (
        f"Block table has {len(block_table)} blocks but need "
        f"{final_blocks_needed} for {final_seq_len} tokens"
    )

    llama_model = worker.model_runner.model.model
    rotary_emb = llama_model.layers[0].self_attn.rotary_emb
    cos_sin_cache = _get_rotary_cos_sin_cache(rotary_emb)
    # Most vLLM rotary_emb expose is_neox_style; Phi3LongRoPE hardcodes neox
    # in its forward and doesn't set the attr — default True for those.
    is_neox = getattr(rotary_emb, 'is_neox_style', True)

    timings = {}

    # ------------------------------------------------------------------
    # Step 1: Prefill B' independently (contiguous FlashAttention)
    # B' sees only its own tokens. Returns contiguous K,V per layer.
    # Memory: O(layers × B'_len × kv_heads × head_dim) — small.
    # ------------------------------------------------------------------
    torch.cuda.synchronize()
    t_prefill = time.perf_counter()

    b_keys, b_values = _prefill_independent_contiguous(
        worker, b_prime_token_ids, head_dim)

    torch.cuda.synchronize()
    timings['prefill_ms'] = round((time.perf_counter() - t_prefill) * 1000, 1)

    # ------------------------------------------------------------------
    # Step 2: Shift C in-place (reverse order for rightward shift)
    # C: [insert_pos, ac_seq_len) → [insert_pos + B'_len, final_seq_len)
    # ------------------------------------------------------------------
    torch.cuda.synchronize()
    t_shift = time.perf_counter()

    if c_len > 0 and b_prime_len > 0:
        c_new_start = insert_pos + b_prime_len
        # Determine iteration direction. Rightward shifts (dst > src) must go
        # reverse to avoid overwriting src before read. Leftward or equal use
        # forward. (Using torch vectorised reads-all-then-writes-all semantics,
        # the direction only matters when src/dst ranges overlap in the same
        # physical block+offset, which is safer to handle explicitly.)
        if c_new_start >= c_source_start:
            c_src = torch.arange(c_source_start + c_len - 1,
                                 c_source_start - 1, -1,
                                 device=device, dtype=torch.long)
        else:
            c_src = torch.arange(c_source_start, c_source_start + c_len,
                                 device=device, dtype=torch.long)
        c_dst = c_src + (c_new_start - c_source_start)

        src_blk = bt_t[c_src // block_size]
        src_off = c_src % block_size
        dst_blk = bt_t[c_dst // block_size]
        dst_off = c_dst % block_size

        for li in range(num_layers):
            kv = kv_caches[li]
            kv[0][dst_blk, dst_off] = kv[0][src_blk, src_off]
            kv[1][dst_blk, dst_off] = kv[1][src_blk, src_off]

    torch.cuda.synchronize()
    timings['shift_ms'] = round((time.perf_counter() - t_shift) * 1000, 1)

    # ------------------------------------------------------------------
    # Step 3: Scatter B' K,V into paged cache at final positions
    # ------------------------------------------------------------------
    torch.cuda.synchronize()
    t_scatter = time.perf_counter()

    b_pos = torch.arange(
        insert_pos, insert_pos + b_prime_len, device=device, dtype=torch.long)
    b_blk = bt_t[b_pos // block_size]
    b_off = b_pos % block_size

    for li in range(num_layers):
        kv_caches[li][0][b_blk, b_off] = b_keys[li]
        kv_caches[li][1][b_blk, b_off] = b_values[li]

    torch.cuda.synchronize()
    timings['scatter_ms'] = round((time.perf_counter() - t_scatter) * 1000, 1)

    # ------------------------------------------------------------------
    # Step 4: RoPE-correct B' and C keys
    # B' prefilled at [0..B'_len) → final [insert_pos..insert_pos+B'_len)
    # C was at [insert_pos..insert_pos+C_len) → [insert_pos+B'_len..final)
    # ------------------------------------------------------------------
    torch.cuda.synchronize()
    t_rope = time.perf_counter()

    # B' RoPE: undo pos [0..B'), redo pos [insert_pos..insert_pos+B')
    b_old_pos = torch.arange(b_prime_len, device=device, dtype=torch.long)
    b_new_pos = b_old_pos + insert_pos
    for li in range(num_layers):
        k = kv_caches[li][0][b_blk, b_off]
        kv_caches[li][0][b_blk, b_off] = _rope_correct_keys_via_cache(
            k, b_old_pos, b_new_pos, cos_sin_cache, is_neox)

    # C RoPE: K was rotated at [c_source_rope_start, c_source_rope_start+c_len)
    # in its pre-shift state. After step 2 the data is at
    # [insert_pos+b_prime_len, final_seq_len). Undo old pos, redo new pos.
    # Skip if the shift is a no-op (REPLACE delta=0 case where C didn't move).
    if c_len > 0:
        c_old_pos = torch.arange(
            c_source_rope_start, c_source_rope_start + c_len,
            device=device, dtype=torch.long)
        c_new_pos = torch.arange(
            insert_pos + b_prime_len, insert_pos + b_prime_len + c_len,
            device=device, dtype=torch.long)
        if not torch.equal(c_old_pos, c_new_pos):
            c_blk = bt_t[c_new_pos // block_size]
            c_off = c_new_pos % block_size
            for li in range(num_layers):
                k = kv_caches[li][0][c_blk, c_off]
                kv_caches[li][0][c_blk, c_off] = _rope_correct_keys_via_cache(
                    k, c_old_pos, c_new_pos, cos_sin_cache, is_neox)

    torch.cuda.synchronize()
    timings['rope_ms'] = round((time.perf_counter() - t_rope) * 1000, 1)

    # ------------------------------------------------------------------
    # Step 5: Selective recompute (CacheBlend deviation-select)
    # B' needs A↔B' cross-attention. C needs B'↔C cross-attention.
    # All B' tokens are repair targets (they saw nothing).
    # Top repair_ratio of C tokens are repair targets.
    # ------------------------------------------------------------------
    torch.cuda.synchronize()
    t_repair = time.perf_counter()

    num_repaired = 0
    if repair_ratio > 0 and abc_token_ids is not None:
        from vllm.kvlobotomy_repair import (
            fast_selective_recompute,
            diagnose_attention_to_b,
            select_repair_candidates,
        )

        # All B' tokens need repair (saw no A context during independent prefill)
        b_repair = list(range(b_prime_len))

        # C-selector options:
        #   'tail' (default): last c_ratio tokens of C. Exact bottom-right mask
        #     in prefill FA; covers generation region (CacheBlend suffix-pin).
        #     REMOVE experiment proved tail beats boundary 97%/64% on
        #     generation-at-tail tasks.
        #   'vdev': V-deviation scattered top-k. Picks the tokens most
        #     changed by the B→B' swap. Needed when answer-carrying content
        #     is in C-middle (multi-hop QA), not tail.
        #   'boundary': first c_ratio tokens of C. Known-bad on
        #     generation-at-tail; exposed here only for ablations.
        #   'none' or ratio=0: skip C repair, repair only B'.
        c_repair_n = max(1, int(c_len * repair_ratio)) if c_len > 0 else 0
        if c_repair_n == 0:
            c_repair = []
        elif c_selector == 'tail':
            c_repair = list(range(
                b_prime_len + c_len - c_repair_n,
                b_prime_len + c_len,
            ))
        elif c_selector == 'boundary':
            c_repair = list(range(
                b_prime_len,
                b_prime_len + c_repair_n,
            ))
        elif c_selector == 'vdev':
            # V-dev needs a diagnostic pass over C vs its stale counterpart.
            # We run diagnose_attention_to_b with delete_start=insert_pos and
            # delete_end=insert_pos+b_prime_len, treating B' as "the region
            # to diagnose against" — V_diff then highlights C tokens whose
            # V values moved most between stale (wrong_B context) and current
            # (B' context). Call ALWAYS requires worker, so caller provides it.
            diag = diagnose_attention_to_b(
                worker, final_seq_len, insert_pos,
                insert_pos + b_prime_len, block_table,
            )
            c_picks = select_repair_candidates(diag, ratio=repair_ratio)
            # c_picks are relative to C (c_picks in [0, c_len)); shift by
            # b_prime_len to put them in the all_repair index space
            # (delete_start=insert_pos, so repair indices are relative to
            # B'+C). Optionally pin last c_pin_last tokens of C.
            pick_set = set(p + b_prime_len for p in c_picks)
            if c_pin_last > 0:
                for j in range(max(0, c_len - c_pin_last), c_len):
                    pick_set.add(b_prime_len + j)
            c_repair = sorted(pick_set)
        elif c_selector == 'none':
            c_repair = []
        else:
            raise ValueError(f'unknown c_selector: {c_selector}')

        all_repair = b_repair + c_repair
        num_repaired = len(all_repair)

        # Two-call FA refactor: when c_selector='tail' (or no c_repair), split
        # repair into (1) B' tokens scoped to A+B' and (2) C-tail tokens scoped
        # to full sequence. Both calls use prefill-mode FA (packed bottom-right
        # mask) because queries ARE at the tail of their respective scopes.
        # This avoids the decode-mode cost (9× slower per measurement) without
        # sacrificing correctness. For vdev/boundary selectors (scattered in C),
        # fall back to single-call decode mode where per-query masks are exact.
        # Always split B' repair from C repair so B' gets the fast prefill path
        # (B' IS the tail of A+B', so bottom-right mask is exact).
        # C repair path depends on selector:
        #   - tail/none → prefill mode (C-tail is at sequence end)
        #   - vdev/boundary scattered → caller's attn_mode (default 'decode')
        if b_prime_len > 0 and b_repair:
            fast_selective_recompute(
                worker=worker,
                new_seq_len=insert_pos + b_prime_len,
                block_table=block_table,
                repair_indices=b_repair,
                head_dim=head_dim,
                delete_start=insert_pos,
                ac_token_ids=list(abc_token_ids),
                _attn_mode='prefill',
            )
        if c_repair:
            if c_selector in ('tail', 'none') and repair_attn_mode == 'decode':
                _c_mode = 'prefill'
            else:
                _c_mode = repair_attn_mode
            fast_selective_recompute(
                worker=worker,
                new_seq_len=final_seq_len,
                block_table=block_table,
                repair_indices=c_repair,
                head_dim=head_dim,
                delete_start=insert_pos,
                ac_token_ids=list(abc_token_ids),
                _attn_mode=_c_mode,
            )

    torch.cuda.synchronize()
    timings['repair_ms'] = round((time.perf_counter() - t_repair) * 1000, 1)

    total_ms = (time.perf_counter() - t0) * 1000

    return {
        **timings,
        'total_ms': round(total_ms, 1),
        'b_prime_len': b_prime_len,
        'c_len': c_len,
        'final_seq_len': final_seq_len,
        'num_repaired': num_repaired,
    }


def fast_replace(worker, total_seq_len, block_table, b_prime_token_ids,
                 delete_start, delete_end, head_dim, rope_theta,
                 num_kv_heads, abc_token_ids=None, repair_ratio=0.15,
                 repair_attn_mode='decode',
                 c_selector='tail', c_pin_last=0):
    """Replace B with B' in [A,B,C] via independent B' prefill (Path 2).

    Memory-efficient: no C extraction buffer.

    Steps:
      1. Shift C from [delete_end, total_seq_len) to
         [delete_start + b_prime_len, ...) — in-place
      2. Prefill B' independently (contiguous FA)
      3. Scatter B' KV into paged cache
      4. RoPE-correct B' and C
      5. Selective recompute for cross-attention repair

    Args:
        worker: vLLM GPU worker
        total_seq_len: length of [A,B,C] currently in cache
        block_table: physical block table (must cover final length)
        b_prime_token_ids: token IDs for B'
        delete_start: start of old B (inclusive)
        delete_end: end of old B (exclusive)
        head_dim, rope_theta, num_kv_heads: model params
        abc_token_ids: full [A, B', C] token IDs (needed for repair)
        repair_ratio: fraction of tokens to selectively recompute

    Returns:
        dict with per-step timing breakdown
    """
    # REPLACE via fast_insert with c_source_start=delete_end.
    # The cache currently holds [A, wrong_B, C] contiguously, so C lives at
    # [delete_end, delete_end+c_len) with K rotated at those same positions.
    # We tell fast_insert to read C from that source location instead of the
    # default [insert_pos, insert_pos+c_len) (which would be wrong_B's slot).
    # fast_insert then:
    #   (a) shifts C from [delete_end, ...) to [a_len+b_prime_len, ...)
    #   (b) writes B' K/V into [a_len, a_len+b_prime_len)
    #   (c) RoPE-corrects C from original positions to final positions
    # This replaces the previous double-shift bug (fast_replace pre-shifted
    # then fast_insert re-shifted reading garbage).
    a_len = delete_start
    old_b_len = delete_end - delete_start
    c_len = total_seq_len - delete_end
    b_prime_len = len(b_prime_token_ids)
    ac_seq_len = a_len + c_len

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    kv_caches = worker.model_runner.kv_caches
    first_cache = kv_caches[0]
    _, _, block_size, _, _ = first_cache.shape
    device = first_cache.device
    num_layers = len(kv_caches)
    bt_t = torch.tensor(block_table, device=device, dtype=torch.long)

    final_seq_len = a_len + b_prime_len + c_len
    delta = b_prime_len - old_b_len

    result = fast_insert(
        worker=worker,
        ac_seq_len=ac_seq_len,
        block_table=block_table,
        b_prime_token_ids=b_prime_token_ids,
        insert_pos=a_len,
        head_dim=head_dim,
        rope_theta=rope_theta,
        num_kv_heads=num_kv_heads,
        abc_token_ids=abc_token_ids,
        repair_ratio=repair_ratio,
        repair_attn_mode=repair_attn_mode,
        c_selector=c_selector,
        c_pin_last=c_pin_last,
        c_source_start=delete_end,          # C sits after wrong_B
        c_source_rope_start=delete_end,     # and K was rotated there
    )

    # Zero stale tail if final sequence got shorter (old_b_len > b_prime_len)
    if final_seq_len < total_seq_len:
        stale = torch.arange(final_seq_len, total_seq_len,
                             device=device, dtype=torch.long)
        s_log = stale // block_size
        s_off = stale % block_size
        s_blk = bt_t[s_log]
        for layer_idx in range(num_layers):
            kv = kv_caches[layer_idx]
            kv[0][s_blk, s_off] = 0
            kv[1][s_blk, s_off] = 0

    total_ms = (time.perf_counter() - t0) * 1000
    result['total_ms'] = round(total_ms, 1)
    result['old_b_len'] = old_b_len
    result['size_delta'] = delta

    return result
