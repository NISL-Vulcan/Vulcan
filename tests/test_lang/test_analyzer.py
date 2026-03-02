"""Unit tests for vulcan.lang CodeAnalyzer."""
import pytest

try:
    from vulcan.lang.CodeAnalysis.analyzer import CodeAnalyzerFactory
    LANG_AVAILABLE = True
except ImportError:
    LANG_AVAILABLE = False

pytestmark = pytest.mark.skipif(not LANG_AVAILABLE, reason="CodeAnalysis import failed")


def test_create_analyzer_python():
    a = CodeAnalyzerFactory.create_analyzer("python")
    assert a is not None


def test_create_analyzer_c():
    a = CodeAnalyzerFactory.create_analyzer("c")
    assert a is not None


def test_create_analyzer_cpp():
    a = CodeAnalyzerFactory.create_analyzer("cpp")
    assert a is not None


def test_create_analyzer_llvmir():
    a = CodeAnalyzerFactory.create_analyzer("llvmir")
    assert a is not None


def test_create_analyzer_java():
    try:
        a = CodeAnalyzerFactory.create_analyzer("java")
        assert a is not None
    except (ImportError, NameError):
        pytest.skip("java impl not available")


def test_create_analyzer_unknown_raises():
    with pytest.raises(ValueError, match="No analyzer"):
        CodeAnalyzerFactory.create_analyzer("unknown_lang")
