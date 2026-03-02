"""Unit tests for vulcan.framework.utils.training."""
import torch
import pytest

from vulcan.framework.utils.training import (
    segment_sizes_to_slices,
    cut_encoded_contexts,
    cut_gadgets_encoded_contexts,
    cut_sys_encoded_contexts,
)


def test_segment_sizes_to_slices():
    sizes = torch.tensor([2, 3, 1])
    slices = segment_sizes_to_slices(sizes)
    assert len(slices) == 3
    assert slices[0] == slice(0, 2)
    assert slices[1] == slice(2, 5)
    assert slices[2] == slice(5, 6)


def test_cut_encoded_contexts():
    enc = torch.randn(6, 8)
    contexts_per_label = [2, 3, 1]
    batched, mask = cut_encoded_contexts(enc, contexts_per_label)
    assert batched.shape == (3, 3, 8)
    assert mask.shape == (3, 3)
    assert (mask[0, 2:] < 0).all()
    assert (mask[1, 3:] < 0).all()


def test_cut_gadgets_encoded_contexts():
    gadgets = torch.randn(10, 4)
    is_back = [False, True]
    words_per_label = [3, 4]
    sen_len = 5
    batched, mask = cut_gadgets_encoded_contexts(gadgets, is_back, words_per_label, sen_len)
    assert batched.shape == (2, 5, 4)
    assert mask.shape == (2, 5)


def test_cut_sys_encoded_contexts():
    sys = torch.randn(7, 8)
    words_per_label = [2, 5]
    sen_len = 6
    batched, mask = cut_sys_encoded_contexts(sys, words_per_label, sen_len)
    assert batched.shape == (2, 6, 8)
    assert mask.shape == (2, 6)
