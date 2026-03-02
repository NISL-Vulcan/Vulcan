"""Unit tests for vulcan.framework.errors."""
import pytest

from vulcan.framework.errors import (
    DatasetInitError,
    DatasetNotMatch,
    BenchmarkInitError,
    DownloadFailed,
    TooManyRequests,
)


def test_dataset_init_error():
    with pytest.raises(DatasetInitError):
        raise DatasetInitError("test")
    e = DatasetInitError("msg")
    assert isinstance(e, OSError)
    assert "msg" in str(e)


def test_dataset_not_match():
    with pytest.raises(DatasetNotMatch):
        raise DatasetNotMatch("mismatch")
    assert issubclass(DatasetNotMatch, OSError)


def test_benchmark_init_error():
    with pytest.raises(BenchmarkInitError):
        raise BenchmarkInitError("bench")
    e = BenchmarkInitError("msg")
    assert isinstance(e, (OSError, ValueError))


def test_download_failed():
    with pytest.raises(DownloadFailed):
        raise DownloadFailed("network error")
    assert issubclass(DownloadFailed, IOError)


def test_too_many_requests():
    with pytest.raises(TooManyRequests):
        raise TooManyRequests("429")
    assert issubclass(TooManyRequests, DownloadFailed)
