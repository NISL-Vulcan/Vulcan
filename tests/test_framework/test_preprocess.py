"""Unit tests for vulcan.framework.preprocess."""
import torch
import pytest

from vulcan.framework.preprocess import (
    Normalize,
    PadSequence,
    OneHotEncode,
    LengthNormalization,
    Shuffle,
    Compose,
    get_preprocess,
)


def test_normalize_default():
    t = Normalize()
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor(0)
    out_x, out_y = t(x, y)
    torch.testing.assert_close(out_x, x)
    assert out_y is y


def test_normalize_custom_mean_std():
    t = Normalize(mean=2.0, std=2.0)
    x = torch.tensor([2.0, 4.0, 6.0])
    y = torch.tensor(1)
    out_x, out_y = t(x, y)
    torch.testing.assert_close(out_x, torch.tensor([0.0, 1.0, 2.0]))
    assert out_y.item() == 1


def test_pad_sequence_short():
    t = PadSequence(pad_value=0.0, max_length=5)
    x = torch.tensor([1.0, 2.0])
    y = torch.tensor(0)
    out_x, out_y = t(x, y)
    assert out_x.shape == (5,)
    assert out_x[0].item() == 1.0 and out_x[1].item() == 2.0
    assert (out_x[2:] == 0).all()


def test_pad_sequence_long():
    t = PadSequence(max_length=3)
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    out_x, _ = t(x, torch.tensor(0))
    assert out_x.shape == (3,)
    assert out_x[0].item() == 1.0 and out_x[2].item() == 3.0


def test_one_hot_encode_binary():
    t = OneHotEncode(num_classes=2)
    x = torch.tensor([1.0, 2.0])
    out_x, out_y = t(x, 0)
    torch.testing.assert_close(out_y, torch.tensor([1.0, 0.0]))
    out_x, out_y = t(x, 1)
    torch.testing.assert_close(out_y, torch.tensor([0.0, 1.0]))


def test_one_hot_encode_three_classes():
    t = OneHotEncode(num_classes=3)
    _, out_y = t(torch.tensor([1.0]), 2)
    torch.testing.assert_close(out_y, torch.tensor([0.0, 0.0, 1.0]))


def test_length_normalization_short():
    """Short sequence: pad_sequence builds a 2D batch with x and zero padding."""
    t = LengthNormalization(max_length=5)
    x = torch.tensor([1.0, 2.0])
    out_x, out_y = t(x, torch.tensor(0))
    assert out_x.dim() == 2
    assert out_x.size(0) == 2
    assert out_x.size(1) == max(2, 5 - 2)


def test_length_normalization_long():
    """Long sequence: truncate directly into a 1D tensor."""
    t = LengthNormalization(max_length=3)
    x = torch.randn(10)
    out_x, _ = t(x, torch.tensor(0))
    assert out_x.dim() == 1
    assert out_x.size(0) == 3


def test_shuffle_deterministic():
    t = Shuffle(seed=42)
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor(0)
    out1, _ = t(x.clone(), y)
    out2, _ = t(x.clone(), y)
    torch.testing.assert_close(out1, out2)


def test_compose_empty():
    c = Compose([])
    x, y = torch.tensor([1.0]), torch.tensor(0)
    out_x, out_y = c(x, y)
    torch.testing.assert_close(out_x, x)
    assert out_y is y


def test_compose_two():
    c = Compose([Normalize(0.0, 1.0), OneHotEncode(2)])
    x = torch.randn(4)
    out_x, out_y = c(x, 1)
    assert out_x.shape == x.shape
    torch.testing.assert_close(out_y, torch.tensor([0.0, 1.0]))


def test_get_preprocess_none():
    p = get_preprocess(128, None)
    assert isinstance(p, Compose)
    assert len(p.transforms) == 0


def test_get_preprocess_list():
    p = get_preprocess(128, ["Normalize", "PadSequence", "OneHotEncode"])
    assert isinstance(p, Compose)
    assert len(p.transforms) == 3


def test_get_preprocess_unknown_raises():
    with pytest.raises(KeyError):
        get_preprocess(128, ["UnknownTransform"])
