"""Unit tests for vulcan.lang CodeAnalyzer."""
import importlib
import sys
import types

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


def _install_c_family_stubs(monkeypatch):
    fake_antlr4 = types.ModuleType("antlr4")
    fake_antlr4.FileStream = lambda file_path, encoding="utf8": ("stream", file_path, encoding)
    fake_antlr4.CommonTokenStream = lambda lexer: ("tokens", lexer)
    monkeypatch.setitem(sys.modules, "antlr4", fake_antlr4)

    class _Lexer:
        def __init__(self, stream):
            self.stream = stream

    class _Parser:
        def __init__(self, token_stream):
            self.token_stream = token_stream

        def translationunit(self):
            return "parse-tree"

    lexer_mod = types.ModuleType("CPP14_v2Lexer")
    lexer_mod.CPP14_v2Lexer = _Lexer
    parser_mod = types.ModuleType("CPP14_v2Parser")
    parser_mod.CPP14_v2Parser = _Parser
    monkeypatch.setitem(
        sys.modules,
        "vulcan.lang.CodeAnalysis.utils.c_utils.src.antlr.gen.CPP14_v2Lexer",
        lexer_mod,
    )
    monkeypatch.setitem(
        sys.modules,
        "vulcan.lang.CodeAnalysis.utils.c_utils.src.antlr.gen.CPP14_v2Parser",
        parser_mod,
    )

    class _CFGExtractorVisitor:
        def __init__(self):
            self.functions = {"f1": "g1", "f2": "g2"}

        def visit(self, parse_tree):
            return None

    cfg_mod = types.ModuleType("cfg_extractor_visitor")
    cfg_mod.CFGExtractorVisitor = _CFGExtractorVisitor
    monkeypatch.setitem(
        sys.modules,
        "vulcan.lang.CodeAnalysis.utils.c_utils.src.cfg_extractor.cfg_extractor_visitor",
        cfg_mod,
    )

    called = {"draw": 0}

    def _draw_cfg(graph, out, token_stream, verbose=False):
        called["draw"] += 1

    visual_mod = types.ModuleType("visual")
    visual_mod.draw_CFG = _draw_cfg
    monkeypatch.setitem(
        sys.modules,
        "vulcan.lang.CodeAnalysis.utils.c_utils.src.graph.visual",
        visual_mod,
    )

    fake_pg = types.ModuleType("programl")
    fake_pg.from_cpp = lambda code: {"code": code}
    fake_pg.to_dot = lambda g: "dot-cpp"
    monkeypatch.setitem(sys.modules, "programl", fake_pg)
    return called


def test_c_analyzer_methods_with_stubs(monkeypatch):
    from vulcan.lang.CodeAnalysis.implementations.c_impl import CAnalyzer

    called = _install_c_family_stubs(monkeypatch)
    analyzer = CAnalyzer()
    assert analyzer.get_ast("demo.c") == "parse-tree"
    assert analyzer.get_cfg("demo.c") is None
    assert called["draw"] == 2
    assert analyzer.get_pg("int main(){}") == "dot-cpp"


def test_cpp_analyzer_methods_with_stubs(monkeypatch):
    from vulcan.lang.CodeAnalysis.implementations.cpp_impl import CppAnalyzer

    called = _install_c_family_stubs(monkeypatch)
    analyzer = CppAnalyzer()
    assert analyzer.get_ast("demo.cpp") == "parse-tree"
    assert analyzer.get_cfg("demo.cpp") is None
    assert called["draw"] == 2
    assert analyzer.get_pg("int main(){}") == "dot-cpp"


def test_llvmir_analyzer_methods_with_stubs(monkeypatch):
    from vulcan.lang.CodeAnalysis.implementations.llvm_ir_impl import LLVMIRAnalyzer

    fake_antlr4 = types.ModuleType("antlr4")
    fake_antlr4.FileStream = lambda file_path, encoding="utf8": ("stream", file_path, encoding)
    fake_antlr4.CommonTokenStream = lambda lexer: ("tokens", lexer)
    monkeypatch.setitem(sys.modules, "antlr4", fake_antlr4)

    class _Lexer:
        def __init__(self, stream):
            self.stream = stream

    class _Parser:
        def __init__(self, token_stream):
            self.token_stream = token_stream

        def compilationUnit(self):
            return "llvm-tree"

    lexer_mod = types.ModuleType("LLVMIRLexer")
    lexer_mod.LLVMIRLexer = _Lexer
    parser_mod = types.ModuleType("LLVMIRParser")
    parser_mod.LLVMIRParser = _Parser
    monkeypatch.setitem(
        sys.modules,
        "vulcan.lang.CodeAnalysis.utils.llvmir_utils.antlr.gen.LLVMIRLexer",
        lexer_mod,
    )
    monkeypatch.setitem(
        sys.modules,
        "vulcan.lang.CodeAnalysis.utils.llvmir_utils.antlr.gen.LLVMIRParser",
        parser_mod,
    )
    fake_pg = types.ModuleType("programl")
    fake_pg.from_llvm_ir = lambda code: {"ir": code}
    fake_pg.to_dot = lambda g: "dot-llvm"
    monkeypatch.setitem(sys.modules, "programl", fake_pg)

    analyzer = LLVMIRAnalyzer()
    assert analyzer.get_ast("demo.ll") == "llvm-tree"
    assert analyzer.get_cfg("demo.ll") is None
    assert analyzer.get_pg("%1 = add i32 1, 2") == "dot-llvm"


def test_python_analyzer_methods(tmp_path, monkeypatch):
    from vulcan.lang.CodeAnalysis.implementations.python_impl import PythonAnalyzer

    src = tmp_path / "m.py"
    src.write_text(
        "def f1():\n    return 1\n\ndef f2():\n    return 2\n",
        encoding="utf-8",
    )
    analyzer = PythonAnalyzer()
    tree = analyzer.get_ast(str(src))
    assert tree is not None

    calls = []
    monkeypatch.setattr(
        analyzer,
        "_plot_control_flow_graph",
        lambda fn, path: calls.append((fn.__name__, path)),
    )
    analyzer.get_cfg(str(src))
    assert len(calls) == 2
    assert calls[0][0] in {"f1", "f2"}

    with pytest.raises(NameError):
        analyzer.get_pg(str(src))


def test_java_analyzer_methods_with_stubs(monkeypatch, tmp_path):
    fake_antlr4 = types.ModuleType("antlr4")
    fake_antlr4.FileStream = lambda file_path, encoding="utf8": ("stream", file_path, encoding)
    fake_antlr4.CommonTokenStream = lambda lexer: ("tokens", lexer)
    fake_antlr4.StdinStream = object
    monkeypatch.setitem(sys.modules, "antlr4", fake_antlr4)

    class _JavaLexer:
        def __init__(self, stream):
            self.stream = stream

    class _JavaParser:
        def __init__(self, token_stream):
            self.token_stream = token_stream

        def compilationUnit(self):
            return "java-tree"

    lexer_mod = types.ModuleType("JavaLexer")
    lexer_mod.JavaLexer = _JavaLexer
    parser_mod = types.ModuleType("JavaParser")
    parser_mod.JavaParser = _JavaParser
    monkeypatch.setitem(sys.modules, "CodeAnalysis.utils.java_utils.antlr.JavaLexer", lexer_mod)
    monkeypatch.setitem(sys.modules, "CodeAnalysis.utils.java_utils.antlr.JavaParser", parser_mod)

    cfg_mod = types.ModuleType("program_graphs.cfg")
    cfg_mod.parse_java = lambda code: {"cfg": code}
    adg_mod = types.ModuleType("program_graphs.adg")
    adg_mod.parse_java = lambda code: {"adg": code}
    monkeypatch.setitem(sys.modules, "program_graphs.cfg", cfg_mod)
    monkeypatch.setitem(sys.modules, "program_graphs.adg", adg_mod)

    # java_impl uses an absolute import for CodeAnalysis.interfaces.base
    base_mod = importlib.import_module("vulcan.lang.CodeAnalysis.interfaces.base")
    monkeypatch.setitem(sys.modules, "CodeAnalysis.interfaces.base", base_mod)

    sys.modules.pop("vulcan.lang.CodeAnalysis.implementations.java_impl", None)
    java_mod = importlib.import_module("vulcan.lang.CodeAnalysis.implementations.java_impl")
    monkeypatch.setattr(
        java_mod.JavaAnalyzer,
        "get_pg",
        lambda self, code: {"pg": code},
        raising=False,
    )
    try:
        java_mod.JavaAnalyzer.__abstractmethods__ = frozenset()
    except Exception:
        pass
    analyzer = java_mod.JavaAnalyzer()

    src = tmp_path / "Demo.java"
    src.write_text("class Demo {}", encoding="utf-8")
    assert analyzer.get_ast(str(src)) == "java-tree"
    assert analyzer.get_cfg(str(src)) == {"cfg": "class Demo {}"}
    assert analyzer.get_adg("class Demo {}") == {"adg": "class Demo {}"}
