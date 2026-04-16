# SPDX-License-Identifier: Apache-2.0
"""KVLobotomy operations: segment deletion with RoPE correction.

This module implements the delete() operation for both pre-RoPE and post-RoPE
storage modes. It is designed to be dispatched to the GPU worker via
LLM.collective_rpc().

Usage from experiment script:
    import os
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    results = llm.collective_rpc(
        kvlobotomy_delete,
        args=(delete_start, delete_end, rope_storage, rope_theta, head_dim,
              num_kv_heads, total_seq_len),
    )
"""

import time

import torch


def apply_delta_rotation(
    keys: torch.Tensor,
    positions: torch.Tensor,
    delta: int,
    head_dim: int,
    rope_theta: float = 500000.0,
    rope_start: int = 0,
    rope_dim: int = None,
) -> torch.Tensor:
    """Apply a delta RoPE rotation to key vectors.

    Uses the group property: R(new_pos) = R(delta) * R(old_pos)
    So K_corrected = R(delta) * K_old.

    Args:
        keys: [N, num_kv_heads, head_dim] tensor of rotated keys
        positions: [N] tensor of original positions (unused for delta, kept
                   for API consistency)
        delta: position shift (typically negative, e.g., -len(B))
        head_dim: dimension of each attention head
        rope_theta: RoPE base frequency
        rope_start: offset into the last dim at which the rotated portion
                    begins. 0 for standard LLaMA/Mistral/etc. Non-zero for
                    MLA (DeepSeek-V2/V3), where only [kv_lora_rank:kv_lora_rank
                    + qk_rope_head_dim] has RoPE applied.
        rope_dim: number of dims actually rotated, starting at rope_start.
                  None means "use head_dim" (standard models). For MLA this is
                  qk_rope_head_dim (typically 64).
    Returns:
        Corrected keys tensor [N, num_kv_heads, head_dim]
    """
    device = keys.device
    dtype = keys.dtype
    num_tokens, num_heads, d = keys.shape

    if rope_dim is None:
        rope_dim = head_dim

    # Compute rotation frequencies over the rotated slice (not the full d)
    freq_indices = torch.arange(0, rope_dim // 2, device=device, dtype=torch.float32)
    freqs = 1.0 / (rope_theta ** (2.0 * freq_indices / rope_dim))

    # Delta angles
    angles = delta * freqs  # [rope_dim//2]
    cos_delta = torch.cos(angles).to(dtype)
    sin_delta = torch.sin(angles).to(dtype)

    # Slice out the rope portion; leave the rope-free prefix (MLA latent) alone
    k_rope = keys[..., rope_start:rope_start + rope_dim]  # [N, H, rope_dim]
    k_even = k_rope[..., : rope_dim // 2]
    k_odd = k_rope[..., rope_dim // 2 :]
    new_even = k_even * cos_delta - k_odd * sin_delta
    new_odd = k_even * sin_delta + k_odd * cos_delta
    rotated_rope = torch.cat([new_even, new_odd], dim=-1)

    # Stitch back if partial rope
    if rope_start == 0 and rope_dim == d:
        return rotated_rope
    out = keys.clone()
    out[..., rope_start:rope_start + rope_dim] = rotated_rope
    return out


def apply_rope_rotation(
    keys: torch.Tensor,
    positions: torch.Tensor,
    head_dim: int,
    rope_theta: float = 500000.0,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply or undo RoPE rotation on key vectors.

    Args:
        keys: [N, num_kv_heads, head_dim] tensor
        positions: [N] tensor of position indices
        head_dim: dimension of each attention head
        rope_theta: RoPE base frequency
        inverse: if True, undo the rotation (apply R(-pos))
    Returns:
        Rotated (or unrotated) keys tensor [N, num_kv_heads, head_dim]
    """
    device = keys.device
    dtype = keys.dtype
    d = head_dim

    freq_indices = torch.arange(0, d // 2, device=device, dtype=torch.float32)
    freqs = 1.0 / (rope_theta ** (2.0 * freq_indices / d))  # [d//2]

    # angles: [N, d//2] — per-token, per-dimension
    angles = positions.float().unsqueeze(1) * freqs.unsqueeze(0)  # [N, d//2]
    if inverse:
        angles = -angles

    cos_vals = torch.cos(angles).to(dtype)  # [N, d//2]
    sin_vals = torch.sin(angles).to(dtype)  # [N, d//2]

    # Expand for heads: [N, 1, d//2] broadcasts over num_heads
    cos_vals = cos_vals.unsqueeze(1)
    sin_vals = sin_vals.unsqueeze(1)

    # Neox-style rotation
    k_even = keys[..., : d // 2]
    k_odd = keys[..., d // 2 :]

    new_even = k_even * cos_vals - k_odd * sin_vals
    new_odd = k_even * sin_vals + k_odd * cos_vals

    return torch.cat([new_even, new_odd], dim=-1)


def kvlobotomy_delete(worker, delete_start, delete_end, rope_storage,
                      rope_theta, head_dim, num_kv_heads, total_seq_len):
    """Simulate deletion overhead by undoing and redoing RoPE on ALL cached keys.

    For post-RoPE: performs undo+redo RoPE on every key (A+B+C) across all
    layers. This measures the computational cost of re-encoding positions
    for the entire cache. The cache is NOT modified — original keys are
    written back after the computation.

    For pre-RoPE: no-op. Keys are stored unrotated, so there is no position
    encoding work to simulate.

    This function runs on the GPU worker process via collective_rpc.

    Args:
        worker: vLLM worker object (provides access to kv_caches)
        delete_start: start position of segment B (inclusive) — used for
                      metadata reporting only
        delete_end: end position of segment B (exclusive) — used for
                    metadata reporting only
        rope_storage: "pre" or "post"
        rope_theta: RoPE base frequency (e.g., 500000.0 for Llama 3.2)
        head_dim: attention head dimension
        num_kv_heads: number of KV heads
        total_seq_len: total sequence length in cache

    Returns:
        dict with deletion_latency_ms and metadata
    """
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    if rope_storage == "pre":
        # Pre-RoPE: keys are stored unrotated. No position encoding
        # work needed — cost is O(1).
        pass

    elif rope_storage == "post":
        # Post-RoPE: simulate the cost of undoing and redoing RoPE
        # on ALL keys in the cache across ALL layers.
        # The cache is NOT modified — originals are written back.
        if total_seq_len > 0:
            kv_caches = worker.model_runner.kv_caches
            for layer_idx, kv_cache in enumerate(kv_caches):
                key_cache = kv_cache[0]
                block_size = key_cache.shape[1]

                # ALL positions in the cache
                all_positions = torch.arange(
                    0, total_seq_len,
                    device=key_cache.device,
                    dtype=torch.long,
                )

                logical_blocks = all_positions // block_size
                block_offsets = all_positions % block_size
                physical_blocks = logical_blocks  # identity for single-request

                # Read ALL keys (this is a copy via fancy indexing)
                all_keys = key_cache[
                    physical_blocks, block_offsets
                ]  # [total_seq_len, kv_heads, head_dim]

                # Step 1: Undo RoPE — strip position encoding
                raw_keys = apply_rope_rotation(
                    all_keys, all_positions, head_dim, rope_theta,
                    inverse=True,
                )

                # Step 2: Redo RoPE — reapply position encoding
                _ = apply_rope_rotation(
                    raw_keys, all_positions, head_dim, rope_theta,
                    inverse=False,
                )

                # Write ORIGINALS back — cache is unchanged
                key_cache[physical_blocks, block_offsets] = all_keys

    torch.cuda.synchronize()
    end_time = time.perf_counter()
    deletion_latency_ms = (end_time - start_time) * 1000.0

    return {
        "deletion_latency_ms": deletion_latency_ms,
        "rope_storage": rope_storage,
        "delete_start": delete_start,
        "delete_end": delete_end,
        "total_tokens": total_seq_len,
    }


def kvlobotomy_full_delete(worker, delete_start, delete_end, rope_storage,
                            rope_theta, head_dim, num_kv_heads, total_seq_len,
                            block_table=None):
    """Full delete: remove segment B, compact remaining tokens, zero stale tail.

    Unlike kvlobotomy_delete() which only simulates RoPE correction cost, this
    function actually removes B from the KV cache:

    1. Read C+suffix keys/values from their current positions
    2. For post-RoPE: delta-rotate C+suffix keys by -len(B) positions
    3. Write C+suffix to compacted positions (filling B's gap)
    4. Zero out the stale tail (old positions that are now unused)

    For pre-RoPE: step 2 is skipped (keys have no position encoding).

    After this function returns, the caller MUST heal or reset the prefix
    cache hash table to prevent stale hash entries from causing incorrect
    cache hits in subsequent requests.

    Args:
        worker: vLLM worker object (provides access to kv_caches)
        delete_start: start position of segment to delete (inclusive)
        delete_end: end position of segment to delete (exclusive)
        rope_storage: "pre" or "post"
        rope_theta: RoPE base frequency
        head_dim: attention head dimension
        num_kv_heads: number of KV heads
        total_seq_len: total sequence length before deletion
        block_table: list of physical block IDs mapping logical block index
            to physical block in the KV cache tensor. If None, uses identity
            mapping (logical == physical). Must be provided when the physical
            block allocation doesn't start from 0 (which is typical in vLLM
            since block 0 is the null block).

    Returns:
        dict with deletion_latency_ms, rotation_ms, copy_ms, new_seq_len,
        and metadata
    """
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    delete_len = delete_end - delete_start
    tokens_after = total_seq_len - delete_end  # C+suffix tokens to move
    new_seq_len = total_seq_len - delete_len

    rotation_ms = 0.0
    copy_ms = 0.0

    kv_caches = worker.model_runner.kv_caches

    # Build physical block mapping tensor once (shared across layers).
    # The block_table maps logical block index -> physical block ID.
    # If not provided, falls back to identity mapping.
    _bt_tensor = None  # lazily built on first layer

    # Detect MLA layout: kv_cache is a single 3D tensor
    # [num_blocks, block_size, head_size] instead of a tuple of
    # [num_blocks, block_size, kv_heads, head_dim] for key and value.
    _probe = kv_caches[0]
    _is_mla = hasattr(_probe, 'ndim') and _probe.ndim == 3
    # For MLA (DeepSeek-V2/V3): rope only applies to the last qk_rope_head_dim
    # dims of each latent row. The first kv_lora_rank dims are rope-free.
    # Discover these from the attention module.
    _mla_rope_start = 0
    _mla_rope_dim = None
    if _is_mla:
        attn0 = worker.model_runner.model.model.layers[0].self_attn
        _mla_rope_start = getattr(attn0, 'kv_lora_rank', 512)
        _mla_rope_dim = getattr(attn0, 'qk_rope_head_dim', 64)

    for layer_idx, kv_cache in enumerate(kv_caches):
        if _is_mla:
            # MLA: single tensor per layer, shape [num_blocks, block_size, head_size]
            # There is no separate value cache — V is derived at attention time
            # from the compressed latent via kv_b_proj. We shift the latent rows
            # exactly like the standard key cache, then rotate only the rope tail.
            key_cache = kv_cache       # [num_blocks, block_size, head_size]
            val_cache = None            # not used in MLA path
            block_size = key_cache.shape[1]
        else:
            key_cache = kv_cache[0]   # [num_blocks, block_size, kv_heads, head_dim]
            val_cache = kv_cache[1]   # [num_blocks, block_size, kv_heads, head_dim]
            block_size = key_cache.shape[1]

        # Build block table tensor on first layer (same for all layers)
        if _bt_tensor is None and block_table is not None:
            _bt_tensor = torch.tensor(
                block_table, device=key_cache.device, dtype=torch.long,
            )

        torch.cuda.synchronize()
        t_copy_start = time.perf_counter()

        if tokens_after > 0:
            # Source positions (where C+suffix currently are)
            old_positions = torch.arange(
                delete_end, total_seq_len,
                device=key_cache.device, dtype=torch.long,
            )
            # Destination positions (where they go after compaction)
            new_positions = old_positions - delete_len

            # Map logical block indices to physical block IDs
            old_logical = old_positions // block_size
            old_offsets = old_positions % block_size
            new_logical = new_positions // block_size
            new_offsets = new_positions % block_size

            if _bt_tensor is not None:
                old_blocks = _bt_tensor[old_logical]
                new_blocks = _bt_tensor[new_logical]
            else:
                old_blocks = old_logical
                new_blocks = new_logical

            # Gather C+suffix keys and values (fancy indexing = copy)
            c_keys = key_cache[old_blocks, old_offsets]
            # [N, kv_heads, head_dim] for standard; [N, head_size] for MLA
            if val_cache is not None:
                c_vals = val_cache[old_blocks, old_offsets]

            # Post-RoPE: delta-rotate keys to correct positions
            if rope_storage == "post":
                torch.cuda.synchronize()
                t_rot_start = time.perf_counter()

                if _is_mla:
                    # MLA: rotation only over [rope_start:rope_start+rope_dim]
                    # and c_keys is 2D [N, head_size=576]. apply_delta_rotation
                    # expects 3D [N, H, D]; add a fake head dim.
                    c_keys_3d = c_keys.unsqueeze(1)  # [N, 1, 576]
                    c_keys_3d = apply_delta_rotation(
                        c_keys_3d, old_positions, -delete_len,
                        head_dim=c_keys_3d.shape[-1],
                        rope_theta=rope_theta,
                        rope_start=_mla_rope_start,
                        rope_dim=_mla_rope_dim,
                    )
                    c_keys = c_keys_3d.squeeze(1)
                else:
                    c_keys = apply_delta_rotation(
                        c_keys, old_positions, -delete_len, head_dim, rope_theta,
                    )

                torch.cuda.synchronize()
                t_rot_end = time.perf_counter()
                rotation_ms += (t_rot_end - t_rot_start) * 1000.0

            # Scatter to compacted positions
            key_cache[new_blocks, new_offsets] = c_keys
            if val_cache is not None:
                val_cache[new_blocks, new_offsets] = c_vals

        # Zero out stale tail: positions [new_seq_len, total_seq_len)
        # Prevents any residual data from being accessible
        stale_positions = torch.arange(
            new_seq_len, total_seq_len,
            device=key_cache.device, dtype=torch.long,
        )
        if len(stale_positions) > 0:
            stale_logical = stale_positions // block_size
            stale_offsets = stale_positions % block_size
            if _bt_tensor is not None:
                stale_blocks = _bt_tensor[stale_logical]
            else:
                stale_blocks = stale_logical
            key_cache[stale_blocks, stale_offsets] = 0
            if val_cache is not None:
                val_cache[stale_blocks, stale_offsets] = 0

        torch.cuda.synchronize()
        t_copy_end = time.perf_counter()
        copy_ms += (t_copy_end - t_copy_start) * 1000.0

    torch.cuda.synchronize()
    end_time = time.perf_counter()
    deletion_latency_ms = (end_time - start_time) * 1000.0

    # copy_ms includes rotation_ms for post-RoPE; separate them
    copy_only_ms = copy_ms - rotation_ms

    return {
        "deletion_latency_ms": deletion_latency_ms,
        "rotation_ms": rotation_ms,
        "copy_ms": copy_only_ms,
        "total_tokens_moved": tokens_after,
        "total_tokens_deleted": delete_len,
        "new_seq_len": new_seq_len,
        "rope_storage": rope_storage,
        "delete_start": delete_start,
        "delete_end": delete_end,
    }


# ---------------------------------------------------------------------------
# KV cache snapshots (for L2 analysis and visualizer)
# ---------------------------------------------------------------------------

def install_block_capture_hook(worker):
    """Install hook on worker to capture block table after each forward pass.

    The block table maps logical block indices to physical block IDs in the
    KV cache tensor. It's needed by snapshot_kv_cache to read the correct
    physical locations.

    The hook patches execute_model to save the block table from the input
    batch after every forward pass. The last captured table is used by
    snapshot_kv_cache.

    Args:
        worker: vLLM GPU worker (passed by collective_rpc).

    Returns:
        dict with status.
    """
    _original_execute = worker.execute_model

    def _hooked_execute(scheduler_output):
        result = _original_execute(scheduler_output)
        # Capture block table for group 0 (standard attention), request 0
        bt = worker.model_runner.input_batch.block_table[0]
        worker._kvlobotomy_block_table = bt.block_table.np[0].copy()
        return result

    worker.execute_model = _hooked_execute
    return {"status": "hook_installed"}


def snapshot_kv_cache(worker, num_tokens, save_path, metadata):
    """Snapshot the KV cache for the first num_tokens positions.

    Reads key and value tensors from each layer's KV cache using the
    block table captured by install_block_capture_hook. Saves to a .pt
    file compatible with the KV cache visualizer.

    Output format:
        {
            "metadata": {...},
            "layer_0": {"keys": [N, H, D], "values": [N, H, D]},
            "layer_1": ...,
        }

    Args:
        worker: vLLM GPU worker (passed by collective_rpc).
        num_tokens: Number of token positions to snapshot (from position 0).
        save_path: Path to save the .pt file.
        metadata: Dict of metadata to include in the snapshot.

    Returns:
        dict with status, path, num_tokens, num_layers.
    """
    assert hasattr(worker, '_kvlobotomy_block_table'), (
        "No block table captured. Call install_block_capture_hook first "
        "and run at least one forward pass."
    )

    block_ids = worker._kvlobotomy_block_table
    kv_caches = worker.model_runner.kv_caches
    assert len(kv_caches) > 0, "No KV caches found on worker"

    block_size = kv_caches[0][0].shape[1]
    num_blocks_needed = (num_tokens + block_size - 1) // block_size

    snapshot = {"metadata": metadata}

    for layer_idx, kv_cache in enumerate(kv_caches):
        key_cache = kv_cache[0]  # [num_blocks, block_size, kv_heads, head_dim]
        val_cache = kv_cache[1]

        keys_list = []
        vals_list = []
        tokens_read = 0

        for b in range(num_blocks_needed):
            if tokens_read >= num_tokens:
                break
            phys_block = int(block_ids[b])
            remaining = min(block_size, num_tokens - tokens_read)
            keys_list.append(key_cache[phys_block, :remaining].cpu())
            vals_list.append(val_cache[phys_block, :remaining].cpu())
            tokens_read += remaining

        snapshot[f"layer_{layer_idx}"] = {
            "keys": torch.cat(keys_list, dim=0),
            "values": torch.cat(vals_list, dim=0),
        }

    torch.save(snapshot, save_path)
    return {
        "status": "saved",
        "path": save_path,
        "num_tokens": num_tokens,
        "num_layers": len(kv_caches),
    }
