"""Unit tests for vulcan.framework.representations.vectorizers."""
import importlib
import sys
import types

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


def _import_representation_module_with_stubs(module_name, monkeypatch):
    fake_extractors = types.ModuleType("vulcan.framework.representations.extractors.extractors")

    class _Visitor:
        def __init__(self):
            pass

    class _ClangDriver:
        class ProgrammingLanguage:
            C = "c"

        class OptimizationLevel:
            O3 = "o3"

        class IncludeDirType:
            User = "user"

        def __init__(self, *args, **kwargs):
            self._filename = ""

        def getFileName(self):
            return self._filename

        def setFileName(self, name):
            self._filename = name

        def addIncludeDir(self, _path, _kind):
            return None

        def removeIncludeDir(self, _path, _kind):
            return None

    class _Extractor:
        def __init__(self, *args, **kwargs):
            pass

    class _ClangFunctionInfo:
        def __init__(self, name="fn", args=None, entryStmt=None, cfgBlocks=None):
            self.name = name
            self.args = args or []
            self.entryStmt = entryStmt
            self.cfgBlocks = cfgBlocks or []

    class _ClangStmtInfo:
        def __init__(self, name="stmt", ast_relations=None, ref_relations=None):
            self.name = name
            self.ast_relations = ast_relations or []
            self.ref_relations = ref_relations or []

    class _ClangCFGBlock:
        def __init__(self, successors=None, statements=None):
            self.successors = successors or []
            self.statements = statements or []

    fake_extractors.Visitor = _Visitor
    fake_extractors.ClangDriver = _ClangDriver
    fake_extractors.ClangExtractor = _Extractor
    fake_extractors.LLVMIRExtractor = _Extractor
    fake_extractors.LLVM_VERSION = "0.0"
    fake_extractors.clang = types.SimpleNamespace(
        graph=types.SimpleNamespace(
            FunctionInfo=_ClangFunctionInfo,
            StmtInfo=_ClangStmtInfo,
            CfgBlock=_ClangCFGBlock,
        ),
        seq=types.SimpleNamespace(
            TokenInfo=type("TokenInfo", (), {}),
        ),
    )

    class _ArgInfo:
        def __init__(self, type_="i32"):
            self.type = type_

    class _InstructionInfo:
        def __init__(self, opcode="op", operands=None, function=None, call_target=None, type_="i32"):
            self.opcode = opcode
            self.operands = operands or []
            self.function = function
            self.callTarget = call_target
            self.type = type_

    class _BasicBlockInfo:
        def __init__(self, instructions=None, successors=None):
            self.instructions = instructions or []
            self.successors = successors or []

    class _FunctionInfo:
        def __init__(
            self,
            name="fn",
            args=None,
            memory_accesses=None,
            entry_instruction=None,
            exit_instructions=None,
        ):
            self.name = name
            self.args = args or []
            self.memoryAccesses = memory_accesses or []
            self.entryInstruction = entry_instruction
            self.exitInstructions = exit_instructions or []

    class _ConstantInfo:
        def __init__(self, type_="i32"):
            self.type = type_

    fake_extractors.llvm = types.SimpleNamespace(
        graph=types.SimpleNamespace(
            FunctionInfo=_FunctionInfo,
            BasicBlockInfo=_BasicBlockInfo,
            InstructionInfo=_InstructionInfo,
            ArgInfo=_ArgInfo,
            ConstantInfo=_ConstantInfo,
        ),
        seq=types.SimpleNamespace(
            FunctionInfo=type("FunctionInfo", (), {}),
            BasicBlockInfo=type("BasicBlockInfo", (), {}),
            InstructionInfo=type("InstructionInfo", (), {}),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "vulcan.framework.representations.extractors.extractors",
        fake_extractors,
    )
    module = importlib.import_module(module_name)
    return importlib.reload(module)


def test_ast_graphs_filter_and_token_edges(monkeypatch):
    ast_mod = _import_representation_module_with_stubs(
        "vulcan.framework.representations.ast_graphs", monkeypatch
    )
    assert ast_mod.filter_type("int x") == "intType"
    assert ast_mod.filter_type("float y") == "floatType"
    assert ast_mod.filter_type("char a[8]") == "arrayType"
    assert ast_mod.filter_type("f(int)") == "fnType"
    assert ast_mod.filter_type("char") == "type"

    g = ast_mod.nx.MultiDiGraph()
    class _Token:
        def __init__(self, name, index):
            self.name = name
            self.index = index

    class _Node:
        def __init__(self, tokens):
            self.tokens = tokens

    node = _Node(tokens=[_Token("tokA", 0), _Token("tokB", 1)])
    ast_mod.add_token_ast_edges(g, node)
    assert g.number_of_nodes() == 3
    assert g.number_of_edges() == 2


def test_ast_graphs_ast_ref_cfg_edges_and_visitors(monkeypatch):
    ast_mod = _import_representation_module_with_stubs(
        "vulcan.framework.representations.ast_graphs", monkeypatch
    )
    g = ast_mod.nx.MultiDiGraph()

    # Function-level AST edges

    class _Arg:
        def __init__(self, type_, name):
            self.type = type_
            self.name = name

    arg = _Arg(type_="int", name="x")
    entry_stmt = ast_mod.clang.graph.StmtInfo(name="entry")
    func = ast_mod.clang.graph.FunctionInfo(
        name="f",
        args=[arg],
        entryStmt=entry_stmt,
        cfgBlocks=[],
    )
    ast_mod.add_ast_edges(g, func)
    assert any(d.get("attr") == "function" for _, d in g.nodes(data=True))
    assert any(
        u is func and d.get("attr") == "ast" for u, _, d in g.edges(data=True)
    )

    # Statement-level AST and data edges

    class _Ref:
        def __init__(self, type_, name):
            self.type = type_
            self.name = name

    child_stmt = ast_mod.clang.graph.StmtInfo(name="child")
    ref_target = _Ref(type_="float", name="y")
    stmt = ast_mod.clang.graph.StmtInfo(
        name="parent",
        ast_relations=[child_stmt],
        ref_relations=[ref_target],
    )
    ast_mod.add_ast_edges(g, stmt)
    ast_mod.add_ref_edges(g, stmt)
    assert any(
        u is stmt and d.get("attr") == "ast" for u, _, d in g.edges(data=True)
    )
    assert any(d.get("attr") == "data" for _, _, d in g.edges(data=True))

    # CFG edges
    stmt_in_bb = ast_mod.clang.graph.StmtInfo(name="bb_stmt")
    cfg_block_succ = ast_mod.clang.graph.CfgBlock()
    cfg_block = ast_mod.clang.graph.CfgBlock(
        successors=[cfg_block_succ],
        statements=[stmt_in_bb],
    )
    func_cfg = ast_mod.clang.graph.FunctionInfo(
        name="g",
        args=[],
        entryStmt=entry_stmt,
        cfgBlocks=[cfg_block],
    )
    ast_mod.add_cfg_edges(g, func_cfg)
    assert any(d.get("attr") == "cfg" for _, _, d in g.edges(data=True))
    assert any(d.get("attr") == "in" for _, _, d in g.edges(data=True))

    # Visitors and builder
    ast_vis = ast_mod.ASTVisitor()
    ast_vis.visit(func)
    ast_vis.visit(stmt)
    assert any(d.get("attr") == "ast" for _, _, d in ast_vis.G.edges(data=True))

    data_vis = ast_mod.ASTDataVisitor()
    data_vis.visit(stmt)
    assert any(d.get("attr") == "data" for _, _, d in data_vis.G.edges(data=True))

    data_cfg_tok_vis = ast_mod.ASTDataCFGTokenVisitor()

    class _Tok:
        def __init__(self, name, index):
            self.name = name
            self.index = index

    stmt_in_bb.tokens = [_Tok(name="tok", index=0)]
    data_cfg_tok_vis.visit(func_cfg)
    data_cfg_tok_vis.visit(stmt_in_bb)
    assert any(d.get("attr") == "cfg" for _, _, d in data_cfg_tok_vis.G.edges(data=True))
    assert any(d.get("attr") == "token" for _, _, d in data_cfg_tok_vis.G.edges(data=True))

    builder = ast_mod.ASTGraphBuilder(clang_driver=object())

    class _Info:
        @staticmethod
        def accept(visitor):
            visitor.visit(func_cfg)
            visitor.visit(stmt_in_bb)

    graph = builder.info_to_representation(
        _Info(), visitor=ast_mod.ASTDataCFGTokenVisitor
    )
    assert graph.size() > 0
    assert "function" in graph.get_node_str_list()


def test_llvm_seq_transform_helpers(monkeypatch):
    llvm_mod = _import_representation_module_with_stubs(
        "vulcan.framework.representations.llvm_seq", monkeypatch
    )
    assert llvm_mod.merge_after_element_on_condition(["a", "b", "c", "a", "e"], ["a"]) == [
        "ab",
        "c",
        "ae",
    ]
    assert llvm_mod.filer_elements(["a", " ", "c"], [" "]) == ["a", "c"]
    assert llvm_mod.strip_elements(["a", " b", "c\n"], [" ", "\n"]) == ["a", "b", "c"]
    assert llvm_mod.strip_function_name(["@", "foo", "x"]) == ["@", "fn_0", "x"]
    assert llvm_mod.transform_elements(["%", "1", " ", "local_unnamed_addr", "\n", "i", "32"]) == [
        "%1",
        "i32",
    ]


def test_llvm_seq_visitor_collects_tokens(monkeypatch):
    llvm_mod = _import_representation_module_with_stubs(
        "vulcan.framework.representations.llvm_seq", monkeypatch
    )
    vis = llvm_mod.LLVMSeqVisitor()

    fn = llvm_mod.llvm.seq.FunctionInfo()
    fn.signature = ["@", "foo", " ", "i", "32"]
    vis.visit(fn)

    bb = llvm_mod.llvm.seq.BasicBlockInfo()
    bb.name = "bb0"
    vis.visit(bb)

    ins = llvm_mod.llvm.seq.InstructionInfo()
    ins.tokens = ["%", "1", " ", "add"]
    vis.visit(ins)

    assert "fn_0" in vis.S
    assert "bb0:" in vis.S
    assert "%1" in vis.S


def test_syntax_seq_tokenkind_variable_visitor(monkeypatch):
    syn_mod = _import_representation_module_with_stubs(
        "vulcan.framework.representations.syntax_seq", monkeypatch
    )
    vis = syn_mod.SyntaxTokenkindVariableVisitor()

    def _tok(name, kind):
        t = syn_mod.clang.seq.TokenInfo()
        t.name = name
        t.kind = kind
        return t

    vis.visit(_tok("var_1", "raw_identifier"))
    vis.visit(_tok("for", "keyword"))
    vis.visit(_tok("int32_t", "identifier"))
    vis.visit(_tok("other_name", "raw_identifier"))

    assert vis.S == ["var_1", "for", "int32_t", "raw_identifier"]


def test_syntax_seq_visitors_and_builder(monkeypatch):
    syn_mod = _import_representation_module_with_stubs(
        "vulcan.framework.representations.syntax_seq", monkeypatch
    )

    # Prepare tokens to trigger all branch logic
    Tok = syn_mod.clang.seq.TokenInfo

    raw_var = Tok()
    raw_var.name = "var_x"
    raw_var.kind = "raw_identifier"

    kw_for = Tok()
    kw_for.name = "for"
    kw_for.kind = "keyword"

    fn_token = Tok()
    fn_token.name = "fn_0"
    fn_token.kind = "identifier"

    int_token = Tok()
    int_token.name = "int64_t"
    int_token.kind = "identifier"

    float_token = Tok()
    float_token.name = "float32_t"
    float_token.kind = "identifier"

    other = Tok()
    other.name = "x"
    other.kind = "identifier"

    tokens = [raw_var, kw_for, fn_token, int_token, float_token, other]

    # SyntaxSeqVisitor: collect name
    seq_vis = syn_mod.SyntaxSeqVisitor()
    for t in tokens:
        seq_vis.visit(t)
    assert seq_vis.S == [t.name for t in tokens]

    # SyntaxTokenkindVisitor: collect kind
    kind_vis = syn_mod.SyntaxTokenkindVisitor()
    for t in tokens:
        kind_vis.visit(t)
    assert kind_vis.S == [t.kind for t in tokens]

    # SyntaxTokenkindVariableVisitor: exercise all conditional branches
    var_vis = syn_mod.SyntaxTokenkindVariableVisitor()
    for t in tokens:
        var_vis.visit(t)
    assert var_vis.S == [
        "var_x",       # raw_identifier containing "var"
        "for",         # whitelisted keyword
        "fn_0",        # aliased function name
        "int64_t",     # int* prefix
        "float32_t",   # float* prefix
        "identifier",  # others fall back to kind
    ]

    # Builder + Representation: use SyntaxTokenkindVariableVisitor to aggregate token frequencies
    class _Info:
        @staticmethod
        def accept(visitor):
            for t in tokens:
                visitor.visit(t)

    builder = syn_mod.SyntaxSeqBuilder(clang_driver=object())
    seq = builder.info_to_representation(_Info(), visitor=syn_mod.SyntaxTokenkindVariableVisitor)

    assert isinstance(seq, syn_mod.common.Sequence)
    assert seq.S == var_vis.S
    token_list = seq.get_token_list()
    assert len(token_list) == len(seq.S)


def test_llvm_seq_builder_info_to_representation(monkeypatch):
    llvm_mod = _import_representation_module_with_stubs(
        "vulcan.framework.representations.llvm_seq", monkeypatch
    )

    fn = llvm_mod.llvm.seq.FunctionInfo()
    fn.signature = ["@", "foo", " ", "i", "32"]

    bb = llvm_mod.llvm.seq.BasicBlockInfo()
    bb.name = "bb0"

    ins = llvm_mod.llvm.seq.InstructionInfo()
    ins.tokens = ["%", "1", " ", "add"]

    class _Info:
        @staticmethod
        def accept(visitor):
            visitor.visit(fn)
            visitor.visit(bb)
            visitor.visit(ins)

    builder = llvm_mod.LLVMSeqBuilder(clang_driver=object())
    seq = builder.info_to_representation(_Info(), visitor=llvm_mod.LLVMSeqVisitor)

    assert isinstance(seq, llvm_mod.common.Sequence)
    assert "fn_0" in seq.S
    assert "bb0:" in seq.S
    assert "%1" in seq.S
    token_ids = seq.get_token_list()
    assert len(token_ids) == len(seq.S)


def test_llvm_graphs_visitors_and_builder(monkeypatch):
    llvm_graphs_mod = _import_representation_module_with_stubs(
        "vulcan.framework.representations.llvm_graphs", monkeypatch
    )
    llvm = llvm_graphs_mod.llvm

    arg = llvm.graph.ArgInfo("i32")
    dep_inst = llvm.graph.InstructionInfo(opcode="dep")
    mem_inst = llvm.graph.InstructionInfo(opcode="mem")
    memacc = types.SimpleNamespace(inst=mem_inst, dependencies=[types.SimpleNamespace(inst=dep_inst)])

    entry = llvm.graph.InstructionInfo(opcode="entry")
    ret = llvm.graph.InstructionInfo(opcode="ret")
    bb1 = llvm.graph.BasicBlockInfo(instructions=[entry, ret], successors=[])
    callee_entry = llvm.graph.InstructionInfo(opcode="entry")
    callee_exit = llvm.graph.InstructionInfo(opcode="ret")
    bb2 = llvm.graph.BasicBlockInfo(instructions=[callee_entry, callee_exit], successors=[])
    bb1.successors = [bb2]

    callee_fn = llvm.graph.FunctionInfo(
        name="callee",
        args=[arg],
        memory_accesses=[],
        entry_instruction=callee_entry,
        exit_instructions=[callee_exit],
    )
    callee_exit.function = callee_fn
    ret.function = callee_fn
    call_inst = llvm.graph.InstructionInfo(opcode="call", call_target="callee")
    call_inst.operands = [arg, dep_inst]

    cdfg = llvm_graphs_mod.LLVMCDFGVisitor()
    cdfg.visit(llvm.graph.FunctionInfo(name="f", args=[arg], memory_accesses=[memacc], entry_instruction=entry))
    cdfg.visit(bb1)
    cdfg.visit(call_inst)
    assert cdfg.G.has_edge(dep_inst, mem_inst)
    assert any(d["attr"] == "cfg" for _, _, d in cdfg.G.edges(data=True))
    assert any(d["attr"] == "data" for _, _, d in cdfg.G.edges(data=True))

    callv = llvm_graphs_mod.LLVMCDFGCallVisitor()
    callv.visit(callee_fn)
    callv.visit(call_inst)
    callv.visit(ret)
    assert any(d["attr"] == "call" for _, _, d in callv.G.edges(data=True))

    plus = llvm_graphs_mod.LLVMCDFGPlusVisitor()
    plus.visit(llvm.graph.FunctionInfo(name="f2", args=[arg], memory_accesses=[], entry_instruction=entry))
    plus.visit(bb1)
    assert any(d["attr"] == "bb" for _, _, d in plus.G.edges(data=True))

    programl = llvm_graphs_mod.LLVMProGraMLVisitor()
    const = llvm.graph.ConstantInfo("i1")
    inst_operand = llvm.graph.InstructionInfo(opcode="phi", type_="i32")
    pinst = llvm.graph.InstructionInfo(opcode="add", operands=[const, inst_operand], function=callee_fn)
    programl.visit(callee_fn)
    programl.visit(pinst)
    assert any(d["attr"] == "data" for _, _, d in programl.G.edges(data=True))

    class _Info:
        @staticmethod
        def accept(visitor):
            visitor.visit(callee_fn)
            visitor.visit(callee_entry)
            visitor.visit(callee_exit)
            visitor.visit(entry)
            visitor.visit(ret)
            visitor.visit(dep_inst)
            visitor.visit(bb1)
            visitor.visit(call_inst)

    builder = llvm_graphs_mod.LLVMGraphBuilder(clang_driver=object())
    graph = builder.info_to_representation(_Info(), visitor=llvm_graphs_mod.LLVMCDFGCallVisitor)
    assert graph.size() > 0
    assert "function" in graph.get_node_str_list()


def test_common_graph_map_to_leaves_basic_mapping():
    import networkx as nx
    from vulcan.framework.representations import common as common_mod

    G = nx.MultiDiGraph()
    # inner AST node and two leaf tokens
    inner = "inner"
    leaf0 = "leaf0"
    leaf1 = "leaf1"
    G.add_node(inner, attr="inner")
    G.add_node(leaf0, attr="tok0", seq_order=0)
    G.add_node(leaf1, attr="tok1", seq_order=1)
    # ast: inner -> leaf0/leaf1
    G.add_edge(inner, leaf0, attr="ast")
    G.add_edge(inner, leaf1, attr="ast")
    # data edge: inner points to a data node
    data_node = "data"
    G.add_node(data_node, attr="data")
    G.add_edge(inner, data_node, attr="data")

    graph = common_mod.Graph(G, node_types=["inner", "tok0", "tok1", "data"], edge_types=["ast", "data"])
    mapped = graph.map_to_leaves()

    # Leaf nodes should be sorted and kept by seq_order
    leaves_indices = mapped.get_leaf_node_list()
    assert leaves_indices == sorted(leaves_indices)


def test_common_graph_map_to_leaves_custom_relations_and_idempotent():
    import networkx as nx
    from vulcan.framework.representations import common as common_mod

    G = nx.MultiDiGraph()
    root = "root"
    mid = "mid"
    leaf = "leaf"
    G.add_node(root, attr="root")
    G.add_node(mid, attr="mid")
    G.add_node(leaf, attr="leaf", seq_order=5)

    # Custom parent/child relations connected via custom attr values
    G.add_edge(root, mid, attr="P")   # parent relation
    G.add_edge(mid, leaf, attr="C")   # child relation

    graph = common_mod.Graph(G, node_types=["root", "mid", "leaf"], edge_types=["P", "C"])
    relations = {"parent": {"P"}, "child": {"C"}}
    mapped1 = graph.map_to_leaves(relations)
    mapped2 = mapped1.map_to_leaves(relations)

    # root and mid should both be collapsed onto leaf (idempotent: the second mapping does not change the structure)
    assert mapped1.get_node_str_list() == mapped2.get_node_str_list()
