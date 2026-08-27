# SPDX-License-Identifier: Apache-2.0
"""Tied-embedding vocabulary statistics.

Default path is one GEMM over the full vocabulary. A chunked streaming
path remains for callers that pass a smaller ``chunk_size``. Semantics
follow ``dlm/chatdlm1/vocab_ops.py`` (sample, greedy, entropy, confidence)
with Gemma-style tanh softcapping.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from vllm_metal.modilify.config import VOCAB_CHUNK_SIZE

_F32_TINY = 1.1754943508222875e-38
_F32_EPS = 1.1920928955078125e-07
# Oneshot scores of this many fp32 elements fit (~2 GiB); B=8 was proven.
_MAX_SCORE_ELEMS = 2048 * 262_144


@dataclass(frozen=True)
class VocabStatistics:
    proposal: mx.array
    proposal_confidence: mx.array
    token_entropy: mx.array
    greedy_proposal: mx.array
    greedy_confidence: mx.array


def _softcap(logits: mx.array, cap: float) -> mx.array:
    return mx.tanh(logits.astype(mx.float32) / cap) * cap


def oneshot_vocab_statistics(
    hidden: mx.array,
    weight: mx.array,
    *,
    temperature: float,
    softcap: float,
) -> VocabStatistics:
    """Single GEMM over the full vocabulary. Same stats as the chunked path."""
    if hidden.ndim != 2 or weight.ndim != 2:
        raise ValueError("Oneshot vocabulary tensors must be matrices.")
    if hidden.shape[1] != weight.shape[1]:
        raise ValueError("Hidden and vocabulary projection dimensions differ.")
    if temperature <= 0 or softcap <= 0:
        raise ValueError("Temperature and logit softcap must be positive.")

    scores = _softcap(hidden @ weight.T, softcap) / float(temperature)
    log_z = mx.logsumexp(scores, axis=-1)
    greedy = mx.argmax(scores, axis=-1)
    greedy_score = mx.max(scores, axis=-1)
    shifted = scores - log_z[:, None]
    entropy = log_z - mx.sum(mx.exp(shifted) * scores, axis=-1)
    proposal = mx.random.categorical(scores, axis=-1)
    selected_score = mx.take_along_axis(scores, proposal[:, None], axis=-1).reshape(
        (hidden.shape[0],)
    )
    confidence = mx.clip(mx.exp(selected_score - log_z), 0.0, 1.0)
    greedy_confidence = mx.clip(mx.exp(greedy_score - log_z), 0.0, 1.0)
    return VocabStatistics(
        proposal=proposal.astype(mx.int32),
        proposal_confidence=confidence,
        token_entropy=entropy,
        greedy_proposal=greedy.astype(mx.int32),
        greedy_confidence=greedy_confidence,
    )


def chunked_vocab_statistics(
    hidden: mx.array,
    weight: mx.array,
    *,
    temperature: float,
    softcap: float,
    chunk_size: int = VOCAB_CHUNK_SIZE,
    gumbel_noise: mx.array | None = None,
) -> VocabStatistics:
    """Project ``hidden @ weight.T`` in chunks and return sample/greedy stats.

    ``hidden`` is ``[rows, hidden]``, ``weight`` is ``[vocab, hidden]`` (tied
    embedding table). Sampling uses Gumbel-max over temperature-scaled scores.
    """
    if hidden.ndim != 2 or weight.ndim != 2:
        raise ValueError("Chunked vocabulary tensors must be matrices.")
    if hidden.shape[1] != weight.shape[1]:
        raise ValueError("Hidden and vocabulary projection dimensions differ.")
    if temperature <= 0 or chunk_size <= 0:
        raise ValueError("Temperature and vocabulary chunk size must be positive.")
    if softcap <= 0:
        raise ValueError("Logit softcap must be positive.")

    rows = hidden.shape[0]
    vocab_size = weight.shape[0]
    neg_inf = mx.array(-3.402823466e38, dtype=mx.float32)
    sample_log_z = mx.full((rows,), neg_inf, dtype=mx.float32)
    best_gumbel = mx.full((rows,), neg_inf, dtype=mx.float32)
    selected_score = mx.zeros((rows,), dtype=mx.float32)
    selected = mx.zeros((rows,), dtype=mx.int32)
    greedy_score = mx.full((rows,), neg_inf, dtype=mx.float32)
    greedy = mx.zeros((rows,), dtype=mx.int32)
    moment_max = mx.full((rows,), neg_inf, dtype=mx.float32)
    moment_sum = mx.zeros((rows,), dtype=mx.float32)
    moment_weighted = mx.zeros((rows,), dtype=mx.float32)

    for start in range(0, vocab_size, int(chunk_size)):
        stop = min(start + int(chunk_size), vocab_size)
        raw = hidden @ weight[start:stop].T
        scores = _softcap(raw, softcap)
        sample_scores = scores / float(temperature)
        sample_log_z = mx.logaddexp(
            sample_log_z, mx.logsumexp(sample_scores, axis=-1)
        )

        if gumbel_noise is None:
            uniform = mx.random.uniform(
                low=_F32_TINY,
                high=1.0 - _F32_EPS,
                shape=sample_scores.shape,
                dtype=mx.float32,
            )
            noise = -mx.log(-mx.log(uniform))
        else:
            noise = gumbel_noise[:, start:stop]
        gumbel_scores = sample_scores + noise
        chunk_best = mx.max(gumbel_scores, axis=-1)
        chunk_index = mx.argmax(gumbel_scores, axis=-1)
        replace_best = chunk_best > best_gumbel
        candidate_score = mx.take_along_axis(
            sample_scores, chunk_index[:, None], axis=-1
        ).reshape((rows,))
        best_gumbel = mx.maximum(best_gumbel, chunk_best)
        selected = mx.where(replace_best, chunk_index + start, selected)
        selected_score = mx.where(replace_best, candidate_score, selected_score)

        chunk_max = mx.max(sample_scores, axis=-1)
        chunk_argmax = mx.argmax(sample_scores, axis=-1)
        replace_greedy = chunk_max > greedy_score
        greedy_score = mx.maximum(greedy_score, chunk_max)
        greedy = mx.where(replace_greedy, chunk_argmax + start, greedy)

        shifted = mx.exp(sample_scores - chunk_max[:, None])
        chunk_sum = mx.sum(shifted, axis=-1)
        chunk_weighted = mx.sum(shifted * sample_scores, axis=-1)
        merged_max = mx.maximum(moment_max, chunk_max)
        previous_scale = mx.exp(moment_max - merged_max)
        chunk_scale = mx.exp(chunk_max - merged_max)
        moment_sum = moment_sum * previous_scale + chunk_sum * chunk_scale
        moment_weighted = (
            moment_weighted * previous_scale + chunk_weighted * chunk_scale
        )
        moment_max = merged_max

    confidence = mx.clip(mx.exp(selected_score - sample_log_z), 0.0, 1.0)
    greedy_confidence = mx.clip(mx.exp(greedy_score - sample_log_z), 0.0, 1.0)
    entropy = sample_log_z - moment_weighted / mx.maximum(moment_sum, _F32_TINY)
    return VocabStatistics(
        proposal=selected.astype(mx.int32),
        proposal_confidence=confidence,
        token_entropy=entropy,
        greedy_proposal=greedy.astype(mx.int32),
        greedy_confidence=greedy_confidence,
    )


def _softmax_statistics(
    logits: mx.array, temperature: float
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    scores = logits.astype(mx.float32) / temperature
    probabilities = mx.softmax(scores, axis=-1, precise=True)
    greedy_proposal = mx.argmax(probabilities, axis=-1)
    greedy_confidence = mx.squeeze(
        mx.take_along_axis(probabilities, greedy_proposal[..., None], axis=-1),
        axis=-1,
    )
    token_entropy = -mx.sum(
        probabilities * mx.log(mx.maximum(probabilities, 1.0e-30)),
        axis=-1,
    )
    return scores, probabilities, greedy_proposal, greedy_confidence, token_entropy


_compiled_softmax_statistics = mx.compile(_softmax_statistics, shapeless=True)


def dense_vocab_statistics(
    hidden: mx.array,
    embed_as_linear,
    *,
    temperature: float,
    softcap: float,
) -> VocabStatistics:
    """Single tied-embed projection + compiled softmax (Mk1-MLX path)."""
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")
    logits = _softcap(embed_as_linear(hidden), softcap)
    try:
        scores, probabilities, greedy_proposal, greedy_confidence, token_entropy = (
            _compiled_softmax_statistics(logits, temperature)
        )
    except ValueError:
        scores, probabilities, greedy_proposal, greedy_confidence, token_entropy = (
            _softmax_statistics(logits, temperature)
        )
    proposal = mx.random.categorical(scores, axis=-1)
    proposal_confidence = mx.squeeze(
        mx.take_along_axis(probabilities, proposal[..., None], axis=-1),
        axis=-1,
    )
    return VocabStatistics(
        proposal=proposal.astype(mx.int32),
        proposal_confidence=proposal_confidence,
        token_entropy=token_entropy,
        greedy_proposal=greedy_proposal.astype(mx.int32),
        greedy_confidence=greedy_confidence,
    )


def canvas_vocab_statistics(
    hidden: mx.array,
    weight: mx.array,
    *,
    temperature: float,
    softcap: float,
    chunk_size: int = VOCAB_CHUNK_SIZE,
    embed_as_linear=None,
) -> VocabStatistics:
    """``hidden`` is ``[batch, canvas, dim]``; stats keep that layout."""
    if hidden.ndim != 3:
        raise ValueError("Canvas hidden states must have shape [batch, canvas, dim].")
    if embed_as_linear is not None:
        return dense_vocab_statistics(
            hidden,
            embed_as_linear,
            temperature=temperature,
            softcap=softcap,
        )
    batch, canvas, dim = hidden.shape
    flat = hidden.reshape((batch * canvas, dim))
    vocab = int(weight.shape[0])
    rows = int(flat.shape[0])
    if int(chunk_size) >= vocab and rows * vocab <= _MAX_SCORE_ELEMS:
        stats = oneshot_vocab_statistics(
            flat,
            weight,
            temperature=temperature,
            softcap=softcap,
        )
    else:
        safe_chunk = max(1024, min(int(chunk_size), vocab, _MAX_SCORE_ELEMS // max(rows, 1)))
        stats = chunked_vocab_statistics(
            flat,
            weight,
            temperature=temperature,
            softcap=softcap,
            chunk_size=safe_chunk,
        )
    return VocabStatistics(
        proposal=stats.proposal.reshape((batch, canvas)),
        proposal_confidence=stats.proposal_confidence.reshape((batch, canvas)),
        token_entropy=stats.token_entropy.reshape((batch, canvas)),
        greedy_proposal=stats.greedy_proposal.reshape((batch, canvas)),
        greedy_confidence=stats.greedy_confidence.reshape((batch, canvas)),
    )
