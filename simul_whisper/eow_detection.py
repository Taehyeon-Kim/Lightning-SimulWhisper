import logging
import time

import mlx.core as mx
import mlx.nn as nn

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
        cif_linear.weight = weights["weight"]
        cif_linear.bias = weights["bias"]
        mx.eval(cif_linear.parameters())

    return cif_linear, always_fire, never_fire


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
    row_sum = mx.sum(A, axis=1, keepdims=True)  # (N,1)
    scale = T[:, None] / row_sum  # (N,1)
    A = A * scale

    # Precompute mask of nonzeros (stable under the update rule)
    nz_mask = A != 0
    nz_counts = mx.sum(nz_mask, axis=1, keepdims=True)  # (N,1)

    # 2) Iterative damping (fixed # of iters to avoid host sync)
    for _ in range(max_iters):
        # rows that need an update this pass (any entry > threshold)
        needs = mx.any(threshold < A, axis=1, keepdims=True)  # (N,1)

        # per-row mean over nonzeros for current A
        row_sum = mx.sum(A, axis=1, keepdims=True)  # (N,1)
        mean = 0.5 * row_sum / nz_counts  # (N,1)

        # candidate update for all rows, applied only where needed
        updated = A * 0.5 + mean * nz_mask
        A = mx.where(needs, updated, A)

    return A


def fire_at_boundary(chunked_encoder_feature: mx.array, cif_linear, force_eval=False):
    t_start = time.time()

    content_mel_len = chunked_encoder_feature.shape[1]  # B, T, D

    t_linear = time.time()
    alphas = cif_linear(chunked_encoder_feature).squeeze(axis=2)  # B, T
    alphas = mx.sigmoid(alphas)
    if force_eval:
        mx.eval(alphas)
    logger.debug(f"[PERF]       CIF linear+sigmoid: {time.time() - t_linear:.4f}s")

    t_resize = time.time()
    decode_length = mx.round(alphas.sum(axis=-1)).astype(mx.int32)
    alphas = resize(alphas, decode_length)
    if force_eval:
        mx.eval(alphas)
    logger.debug(f"[PERF]       CIF resize: {time.time() - t_resize:.4f}s")

    t_rest = time.time()
    alphas = alphas.squeeze(axis=0)  # (T, )
    threshold = 0.999
    integrate = mx.cumsum(alphas[:-1], axis=0)  # ignore the peak value at the end of the content chunk
    exceed_count = integrate[-1] // threshold
    integrate = integrate - exceed_count * 1.0  # minus 1 every time intergrate exceed the threshold

    mask = integrate >= 0
    if not mx.any(mask):
        if force_eval:
            mx.eval(mask)
        logger.debug(f"[PERF]       CIF rest of computation: {time.time() - t_rest:.4f}s")
        return False

    # Find the index of the first True value using argmax.
    first_true_index = mx.argmax(mask)
    if force_eval:
        mx.eval(first_true_index)
    result = first_true_index.item() >= content_mel_len - 2
    logger.debug(f"[PERF]       CIF rest of computation: {time.time() - t_rest:.4f}s")

    return result
