"""Unit tests for vulcan.framework.representations.vectorizers."""
import pandas as pd
import pytest

from vulcan.framework.representations.vectorizers.vectorizer import Vectorizer


def test_vectorizer_name():
    assert Vectorizer.name() == "basic"


def test_vectorizer_init():
    v = Vectorizer(embedding_dim=64, min_count=2, unknown_node="<UNK>")
    assert v.embedding_dim == 64
    assert v.min_count == 2
    assert v.unknown_node == "<UNK>"


def test_vectorizer_vectorize():
    v = Vectorizer(embedding_dim=32, min_count=1, unknown_node="UNK")
    nodes = [["a", "b"], ["c"]]
    df = v.vectorize(nodes)
    assert isinstance(df, pd.DataFrame)
    assert "id" in df.columns and "node" in df.columns and "vector" in df.columns


def test_vectorizer_create_unknown_raises():
    v = Vectorizer(64, 1, "UNK")
    with pytest.raises(ValueError, match="No such vectorizer"):
        v.create("unknown")
