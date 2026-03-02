"""Unit tests for vulcan.framework.utils.vocabulary."""
import tempfile
import pickle
import pytest
from collections import Counter

from vulcan.framework.utils.vocabulary import (
    Vocabulary_token,
    Vocabulary_c2s,
    PAD,
    UNK,
    SOS,
    EOS,
    TOKEN_TO_ID,
    NODE_TO_ID,
)


def test_xfg_vocabulary_build_from_w2v_and_convert(tmp_path, monkeypatch):
    from vulcan.framework.datasets.XFGDataset_utils import vocabulary as xfg_vocab_mod

    class _FakeWv(dict):
        def __init__(self):
            super().__init__({"foo": 0, "bar": 1})

    class _FakeKeyedVectors:
        def __init__(self):
            self.key_to_index = _FakeWv()

        @staticmethod
        def load(path, mmap="r"):
            return _FakeKeyedVectors()

    # Stub exists and KeyedVectors.load
    w2v_path = tmp_path / "dummy.wv"
    w2v_path.write_bytes(b"stub")
    monkeypatch.setattr(xfg_vocab_mod, "exists", lambda p: True)
    monkeypatch.setattr(xfg_vocab_mod, "KeyedVectors", _FakeKeyedVectors)

    vocab = xfg_vocab_mod.Vocabulary.build_from_w2v(str(w2v_path))
    # Special tokens should come first
    assert vocab.convert_token_to_id(xfg_vocab_mod.PAD) == 0
    assert vocab.convert_token_to_id(xfg_vocab_mod.UNK) == 1
    assert vocab.convert_token_to_id(xfg_vocab_mod.MASK) == 2
    # Word2Vec tokens should follow
    ids = vocab.convert_tokens_to_ids(["foo", "bar", "baz"])
    assert ids[0] != ids[1]
    # Unknown tokens should map to UNK
    assert ids[2] == vocab.convert_token_to_id(xfg_vocab_mod.UNK)
    assert vocab.get_pad_id() == vocab.convert_token_to_id(xfg_vocab_mod.PAD)


def test_xfg_vocabulary_dump_and_load_roundtrip(tmp_path):
    from vulcan.framework.datasets.XFGDataset_utils import vocabulary as xfg_vocab_mod

    vocab = xfg_vocab_mod.Vocabulary(token_to_id={"<PAD>": 0, "foo": 1})
    p = tmp_path / "xfg_vocab.pkl"
    vocab.dump_vocabulary(str(p))
    loaded = xfg_vocab_mod.Vocabulary.load_vocabulary(str(p))
    assert loaded.token_to_id == vocab.token_to_id



def test_vocabulary_constants():
    assert PAD == "<PAD>"
    assert UNK == "<UNK>"
    assert SOS == "<SOS>"
    assert EOS == "<EOS>"


def test_vocabulary_token_add_from_counter():
    tok = Vocabulary_token(token_to_id={})
    tok.token_to_id = {}
    tok.add_from_counter("token_to_id", Counter(["a", "b", "a", "c"]), n_most_values=10)
    assert "a" in tok.token_to_id
    assert "b" in tok.token_to_id
    assert "c" in tok.token_to_id


def test_vocabulary_token_add_from_counter_invalid_field():
    tok = Vocabulary_token(token_to_id={})
    with pytest.raises(ValueError, match="such_field"):
        tok.add_from_counter("such_field", Counter(["a"]), n_most_values=10)


def test_vocabulary_token_dump_load(tmp_path):
    tok = Vocabulary_token(token_to_id={"a": 0, "b": 1})
    p = tmp_path / "vocab.pkl"
    tok.dump_vocabulary(str(p))
    assert p.exists()
    loaded = Vocabulary_token.load_vocabulary(str(p))
    assert loaded.token_to_id == tok.token_to_id


def test_vocabulary_token_load_nonexistent():
    with pytest.raises(ValueError, match="Can't find vocabulary"):
        Vocabulary_token.load_vocabulary("/nonexistent/path.pkl")


def test_vocabulary_c2s_dump_load(tmp_path):
    vocab = Vocabulary_c2s(token_to_id={"a": 0}, node_to_id={"x": 0})
    p = tmp_path / "c2s.pkl"
    vocab.dump_vocabulary(str(p))
    loaded = Vocabulary_c2s.load_vocabulary(str(p))
    assert loaded.token_to_id == vocab.token_to_id
    assert loaded.node_to_id == vocab.node_to_id
