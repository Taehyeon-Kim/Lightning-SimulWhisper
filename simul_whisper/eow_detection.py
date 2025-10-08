import mlx.core as mx
import mlx.nn as nn
import numpy as np
import logging

logger = logging.getLogger(__name__)

# code for the end-of-word detection based on the CIF model proposed in Simul-Whisper, converted to MLX

def load_cif(cfg, n_audio_state):
    """cfg: AlignAttConfig, n_audio_state: int"""
    cif_linear = nn.Linear(n_audio_state, 1)
    if cfg.cif_ckpt_path is None or not cfg.cif_ckpt_path:
        if cfg.never_fire:
            never_fire = True
            always_fire = False
        else:
            always_fire = True
            never_fire = False
    else:
        always_fire = False
        never_fire = cfg.never_fire
        weights = mx.load(cfg.cif_ckpt_path)
        cif_linear.weight = weights['weight']
        cif_linear.bias = weights['bias']
        mx.eval(cif_linear.parameters())

    return cif_linear, always_fire, never_fire


# from https://github.com/dqqcasia/mosst/blob/master/fairseq/models/speech_to_text/convtransformer_wav2vec_cif.py
def resize_old(alphas, target_lengths, threshold=0.999):
    """
    alpha in thresh=1.0 | (0.0, +0.21)
    target_lengths: if None, apply round and resize, else apply scaling
    """
    _alphas_np = np.array(alphas, copy=False)
    target_lengths_np = np.array(target_lengths, copy=False)
    
    # sum
    _num = _alphas_np.sum(-1)
    num = target_lengths_np.astype(np.float32)
    
    # scaling
    _alphas_np = _alphas_np * (num / _num)[:, np.newaxis]
    
    # rm attention value that exceeds threashold
    count = 0
    while np.any(_alphas_np > threshold):
        count += 1
        if count > 10:
            break
        
        exceeding_indices = np.where(_alphas_np > threshold)
        rows_to_update = np.unique(exceeding_indices[0])

        for row_idx in rows_to_update:
            row = _alphas_np[row_idx]
            mask = row != 0
            mean = 0.5 * row.sum() / mask.sum()
            _alphas_np[row_idx] = row * 0.5 + mean * mask
            
    return mx.array(_alphas_np)
 
 
 
def resize_old(alphas, target_lengths, threshold: float = 0.999):
    """
    Scale each row of `alphas` so its sum matches `target_lengths`, then
    repeatedly damp elements that exceed `threshold` (up to 10 passes).

    Args:
        alphas: array-like, shape (N, T)
        target_lengths: array-like, shape (N,)
        threshold: float in (0, 1]
    Returns:
        mx.array with same shape as `alphas`
    """
    # to MLX arrays
    _alphas = mx.array(alphas)
    target_lengths = mx.array(target_lengths)

    # sums per row and target totals
    _num = mx.sum(_alphas, axis=-1)                     # (N,)
    num = target_lengths.astype(mx.float32)             # (N,)

    # scaling to match target_lengths
    scale = (num / _num)[:, None]                       # (N,1) for broadcasting
    _alphas = _alphas * scale

    # iterative clamping via damping + mean redistribution (up to 10 iters)
    count = 0
    # Use .item() to get a Python bool from lazy MLX expression
    while mx.any(_alphas > threshold).item():
        count += 1
        if count > 10:
            break

        # rows needing update (any element > threshold)
        rows_to_update = mx.any(_alphas > threshold, axis=1)[:, None]  # (N,1)

        # compute per-row mask & mean over nonzeros
        mask = (_alphas != 0.0)
        row_sums = mx.sum(_alphas, axis=1, keepdims=True)              # (N,1)
        nz_counts = mx.sum(mask, axis=1, keepdims=True)                # (N,1)
        mean = 0.5 * row_sums / nz_counts

        # damp + redistribute only on flagged rows
        updated = _alphas * 0.5 + mean * mask
        _alphas = mx.where(rows_to_update, updated, _alphas)

    return _alphas
 
def resize(alphas, target_lengths, threshold: float = 0.999, max_iters: int = 10):
    """
    MLX-only, vectorized.
    Matches the original algorithm's math:
      1) scale each row to target_lengths
      2) up to max_iters: for rows with any value > threshold,
         set row := 0.5*row + mean_nonzero(row)*0.5 on nonzero entries.
    """
    # MLX arrays
    A = mx.array(alphas)
    T = mx.array(target_lengths, dtype=A.dtype)

    # 1) Scale each row to target_lengths
    row_sum = mx.sum(A, axis=1, keepdims=True)             # (N,1)
    scale = (T[:, None] / row_sum)                          # (N,1)
    A = A * scale

    # Precompute mask of nonzeros (stable under the update rule)
    nz_mask = (A != 0)
    nz_counts = mx.sum(nz_mask, axis=1, keepdims=True)      # (N,1)

    # 2) Iterative damping (fixed # of iters to avoid host sync)
    for _ in range(max_iters):
        # rows that need an update this pass (any entry > threshold)
        needs = mx.any(A > threshold, axis=1, keepdims=True)   # (N,1)

        # per-row mean over nonzeros for current A
        row_sum = mx.sum(A, axis=1, keepdims=True)             # (N,1)
        mean = 0.5 * row_sum / nz_counts                       # (N,1)

        # candidate update for all rows, applied only where needed
        updated = A * 0.5 + mean * nz_mask
        A = mx.where(needs, updated, A)

    return A
 
def fire_at_boundary(chunked_encoder_feature: mx.array, cif_linear, force_eval=False):
    import time
    t_start = time.time()
    
    content_mel_len = chunked_encoder_feature.shape[1] # B, T, D
    
    t_linear = time.time()
    alphas = cif_linear(chunked_encoder_feature).squeeze(axis=2) # B, T
    alphas = mx.sigmoid(alphas)
    if force_eval:
        mx.eval(alphas)
    logger.debug(f"[PERF]       CIF linear+sigmoid: {time.time()-t_linear:.4f}s")
    
    t_resize = time.time()
    decode_length = mx.round(alphas.sum(axis=-1)).astype(mx.int32)
    alphas = resize(alphas, decode_length)
    if force_eval:
        mx.eval(alphas)
    logger.debug(f"[PERF]       CIF resize: {time.time()-t_resize:.4f}s")
    
    t_rest = time.time()
    alphas = alphas.squeeze(axis=0) # (T, )
    threshold = 0.999
    integrate = mx.cumsum(alphas[:-1], axis=0) # ignore the peak value at the end of the content chunk
    exceed_count = integrate[-1] // threshold
    integrate = integrate - exceed_count*1.0 # minus 1 every time intergrate exceed the threshold
    
    mask = integrate >= 0
    if not mx.any(mask):
        if force_eval:
            mx.eval(mask)
        logger.debug(f"[PERF]       CIF rest of computation: {time.time()-t_rest:.4f}s")
        return False

    # Find the index of the first True value using argmax.
    first_true_index = mx.argmax(mask)
    if force_eval:
        mx.eval(first_true_index)
    result = first_true_index.item() >= content_mel_len - 2
    logger.debug(f"[PERF]       CIF rest of computation: {time.time()-t_rest:.4f}s")
    
    return result