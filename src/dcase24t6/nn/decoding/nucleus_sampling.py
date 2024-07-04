#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from typing import NamedTuple

import torch
from torch import Tensor, nn

"""
from torchoutil import (
    generate_square_subsequent_mask,
    indices_to_multihot,
    repeat_interleave_nd,
    tensor_to_lengths,
)
"""
from torchoutil.nn import TensorTo

from dcase24t6.nn.decoding.common import AACDecoder

pylog = logging.getLogger(__name__)


class GenerateOutput(NamedTuple):
    predictions: Tensor  # (bsize, max_best_pred_size)
    log_probs: Tensor  # (bsize,)
    beam_predictions: Tensor  # (bsize, beam_size, max_global_pred_size)
    beam_log_probs: Tensor  # (bsize, beam_size)


@torch.no_grad()
def nucleus_sampling(
    decoder: AACDecoder,
    frame_embs,
    frame_embs_pad_mask,
    vocab_size,
    bos_id,
    eos_is,
    pad_id,
    max_pred_size,
    min_pred_size,
    forbid_rep_mask,
):
    preds = 0

    return preds


def _select_k_next_toks(
    logits_i: Tensor,
    prev_sum_lprobs: Tensor,
    is_first: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    :param logits_i: (beam_size, vocab_size)
    :param prev_sum_lprobs: (beam_size,)
    :param is_first: Indicate if this is the first word predicted to avoid predict the same word at the beginning.
    """
    beam_size, vocab_size = logits_i.shape
    # note: Use TensorTo because torch.log_softmax returns f32 even if the tensor is f16, so we need to cast it to the correct precision level
    log_activation = nn.Sequential(nn.LogSoftmax(dim=1), TensorTo(dtype=logits_i.dtype))

    if is_first:
        logits_i = logits_i[0].unsqueeze(dim=0)
        sum_lprobs = log_activation(logits_i)
        # sum_lprobs shape: (1, vocab_size)
    else:
        prev_sum_lprobs = prev_sum_lprobs.unsqueeze(dim=1).expand(
            beam_size,
            vocab_size,
        )
        lprobs_i = log_activation(logits_i)
        sum_lprobs = prev_sum_lprobs + lprobs_i
        # sum_lprobs shape: (beam_size, vocab_size)

    sum_lprobs_flat = sum_lprobs.view(-1)
    new_sum_lprobs, next_token_idxs_flat = torch.topk(sum_lprobs_flat, beam_size)

    prev_beam_idxs = next_token_idxs_flat.div(
        vocab_size,
        rounding_mode="trunc",
    )
    next_word_idxs = next_token_idxs_flat % vocab_size

    # prev_beam_idxs: shape is (beam_size,), values in [0, beam_size[
    # next_word_idxs: shape is (beam_size,), values in [0, vocab_size[
    # sum_lprobs_selected: shape is (beam_size,), values in ]-inf, 0]

    return prev_beam_idxs, next_word_idxs, new_sum_lprobs
