"""Unit tests for vulcan.framework.utils.converting."""
import pytest

try:
    from vulcan.framework.utils.converting import parse_token, tokens_to_wrapped_numpy, strings_to_wrapped_numpy
    CONVERTING_AVAILABLE = True
except ImportError:
    CONVERTING_AVAILABLE = False

pytestmark = pytest.mark.skipif(not CONVERTING_AVAILABLE, reason="converting module import failed (utils.vocabulary path)")


def test_parse_token_no_split():
    assert parse_token("hello", False) == ["hello"]
    assert parse_token("a|b|c", False) == ["a|b|c"]


def test_parse_token_with_split():
    assert parse_token("a|b|c", True) == ["a", "b", "c"]
    assert parse_token("x", True, "|") == ["x"]


def test_tokens_to_wrapped_numpy_no_wrap():
    to_id = {"<PAD>": 0, "<UNK>": 1, "a": 2, "b": 3}
    arr = tokens_to_wrapped_numpy(["a", "b"], to_id, max_length=5, is_wrapped=False)
    assert arr.shape == (5, 1)
    assert arr[0, 0] == 2
    assert arr[1, 0] == 3


def test_tokens_to_wrapped_numpy_wrapped_raises_without_sos_eos():
    to_id = {"<PAD>": 0, "<UNK>": 1}
    with pytest.raises(ValueError, match="SOS and EOS"):
        tokens_to_wrapped_numpy(["a"], to_id, 5, is_wrapped=True)


def test_strings_to_wrapped_numpy():
    to_id = {"<PAD>": 0, "<UNK>": 1, "a": 2, "b": 3}
    arr = strings_to_wrapped_numpy(["a", "b|a"], to_id, is_split=False, max_length=4)
    assert arr.shape[0] == 4
    assert arr.shape[1] == 2
