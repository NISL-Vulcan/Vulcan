import collections
import networkx as nx
import pytest

import vulcan.framework as framework_root
from vulcan.framework.representations import common
import vulcan.framework.representations.extractors as extractors_pkg


def sample_graph():
    G = nx.MultiDiGraph()
    G.add_node("root1", attr="root")
    for n in ["n1", "n2", "n3", "n4", "n5"]:
        G.add_node(n, attr=n)
    for n in ["n1", "n2", "n3"]:
        G.add_edge("root1", n, attr="child")
    G.add_edge("n4", "n3", attr="parent")
    G.add_edge("n4", "n5", attr="child")
    for l in range(7):
        G.add_node("l" + str(l+1), attr="leaf" + str(l+1), seq_order=l)
    G.add_edge("n1", "l1", attr="token")
    G.add_edge("n1", "l2", attr="token")
    G.add_edge("n2", "l3", attr="token")
    G.add_edge("root1", "l4", attr="token")
    G.add_edge("n4", "l5", attr="token")
    G.add_edge("n4", "l6", attr="token")
    G.add_edge("n4", "l7", attr="token")
    G.add_edge("root1", "n3", attr="rel2")
    G.add_edge("l1", "l2", attr="token_rel")
    G.add_edge("n2", "l2", attr="node_token_rel")
    return common.Graph(
        G,
        list(sorted(set(attr for _, attr in G.nodes(data="attr")))),
        list(sorted(set(attr for _, _, attr in G.edges(data="attr")))),
    )


def test_map_to_leaves():
    graph = sample_graph()
    relations = {'child': {'token', 'child'}, 'parent': {'parent'}}
    leaves_only = graph.map_to_leaves(relations)

    # n5 is kept because it has no child nodes
    assert sorted(leaves_only.get_node_str_list()) == [
        'leaf1', 'leaf2', 'leaf3', 'leaf4', 'leaf5', 'leaf6', 'leaf7', 'n5'
    ]

    # map to leaves should be idempotent
    assert sorted(graph.map_to_leaves(relations).G.edges(data='attr')) == sorted(
        leaves_only.map_to_leaves(relations).G.edges(data='attr')
    )

    edges = sorted(leaves_only.G.edges(data='attr'))
    expected_edges = [
        ('l1', 'l1', 'token'),
        ('l1', 'l2', 'token'),
        ('l3', 'l3', 'token'),
        ('l1', 'l4', 'token'),
        ('l5', 'l5', 'token'),
        ('l5', 'l6', 'token'),
        ('l5', 'l7', 'token'),
        ('l1', 'l1', 'child'),
        ('l1', 'l3', 'child'),
        ('l1', 'l5', 'child'),
        ('l5', 'n5', 'child'),
        ('l1', 'l2', 'token_rel'),
        ('l3', 'l2', 'node_token_rel'),
        ('l1', 'l5', 'rel2'),
        ('l5', 'l5', 'parent'),
    ]
    assert edges == sorted(expected_edges)

    without_parent = graph.map_to_leaves({'child': ['token', 'child']})
    assert sorted(without_parent.get_node_str_list()) == [
        'leaf1', 'leaf2', 'leaf3', 'leaf4', 'leaf5', 'leaf6', 'leaf7', 'n3', 'n5'
    ]


def test_graph_get_node_str_list():
    graph = sample_graph()
    node_strs = graph.get_node_str_list()
    assert isinstance(node_strs, list)
    assert "root" in node_strs
    assert "n1" in node_strs


def test_graph_get_node_list():
    graph = sample_graph()
    node_ints = graph.get_node_list()
    assert isinstance(node_ints, list)
    assert all(isinstance(i, int) for i in node_ints)
    assert len(node_ints) == graph.size()


def test_graph_get_edge_list():
    graph = sample_graph()
    edges = graph.get_edge_list()
    assert isinstance(edges, list)
    for e in edges:
        assert len(e) == 3
        assert isinstance(e[0], int) and isinstance(e[1], int) and isinstance(e[2], int)


def test_graph_size():
    graph = sample_graph()
    assert graph.size() == len(graph.G)


def test_representation_builder():
    from vulcan.framework.representations.common import RepresentationBuilder
    rb = RepresentationBuilder()
    assert rb.num_tokens() == 0
    assert rb.get_tokens() == []


def test_sequence_get_token_list_and_size():
    from vulcan.framework.representations.common import Sequence
    token_types = ["a", "b", "c"]
    seq = Sequence(["a", "b", "a", "c"], token_types)
    assert seq.size() == 4
    assert seq.get_token_list() == [0, 1, 0, 2]


def test_graph_get_leaf_node_list():
    graph = sample_graph()
    leaves = graph.get_leaf_node_list()
    assert len(leaves) == 7
    assert leaves == sorted(leaves)


def test_sequence_draw_without_pygraphviz_raises(monkeypatch):
    from vulcan.framework.representations.common import Sequence

    monkeypatch.setattr(common, "pgv", None)
    seq = Sequence(["a", "b"], ["a", "b"])
    with pytest.raises(ModuleNotFoundError, match="pygraphviz is required"):
        seq.draw()


def test_representation_builder_print_tokens(capsys):
    from vulcan.framework.representations.common import RepresentationBuilder

    rb = RepresentationBuilder()
    rb._tokens = collections.OrderedDict([("A", 2), ("B", 1)])
    rb.print_tokens()
    out = capsys.readouterr().out
    assert "NodeID" in out
    assert "A" in out
    assert "B" in out


def test_graph_draw_with_mocked_agraph(monkeypatch, tmp_path):
    graph = sample_graph()

    class _FakeSubgraph:
        def add_node(self, *args, **kwargs):
            return None

        def add_edge(self, *args, **kwargs):
            return None

    class _FakeAgraph:
        def subgraph(self, *args, **kwargs):
            return _FakeSubgraph()

        def layout(self, prog):
            assert prog == "dot"
            return None

        def draw(self, path):
            return str(path)

    monkeypatch.setattr(
        "vulcan.framework.representations.common.nx.drawing.nx_agraph.to_agraph",
        lambda G: _FakeAgraph(),
    )
    out = graph.draw(path=str(tmp_path / "g.png"), with_legend=True, align_tokens=True)
    assert out.endswith("g.png")


def test_framework_show_helpers_print(capsys):
    framework_root.show_models()
    framework_root.show_representations()
    framework_root.show_losses()
    framework_root.show_datasets()
    out = capsys.readouterr().out
    assert "Model Names" in out
    assert "Representation Names" in out
    assert "Loss Names" in out
    assert "Datasets" in out


def test_extractors_clang_binary_path_no_extension(monkeypatch):
    extractors_pkg.clang_binary_path.cache_clear()
    monkeypatch.setattr(extractors_pkg, "_EXTENSION_AVAILABLE", False)
    with pytest.raises(ModuleNotFoundError, match="native extension"):
        extractors_pkg.clang_binary_path()


def test_extractors_clang_binary_path_exact_match(monkeypatch):
    extractors_pkg.clang_binary_path.cache_clear()
    monkeypatch.setattr(extractors_pkg, "_EXTENSION_AVAILABLE", True)
    monkeypatch.setattr(extractors_pkg, "LLVM_VERSION", "15.0.0")
    monkeypatch.setattr(extractors_pkg.shutil, "which", lambda name: "/usr/bin/clang-15.0.0" if name == "clang-15.0.0" else None)

    class _Res:
        def __init__(self, txt):
            self.stdout = txt.encode()

    def _fake_run(args, check, stdout):
        if args[1] == "-print-prog-name=llvm-config":
            return _Res("/usr/bin/llvm-config\n")
        return _Res("15.0.0\n")

    monkeypatch.setattr(extractors_pkg.subprocess, "run", _fake_run)
    assert extractors_pkg.clang_binary_path() == "/usr/bin/clang-15.0.0"


def test_extractors_clang_binary_path_best_effort_warns(monkeypatch):
    extractors_pkg.clang_binary_path.cache_clear()
    monkeypatch.setattr(extractors_pkg, "_EXTENSION_AVAILABLE", True)
    monkeypatch.setattr(extractors_pkg, "LLVM_VERSION", "15.0.2")
    monkeypatch.setattr(extractors_pkg.shutil, "which", lambda name: "/usr/bin/clang-15" if name == "clang-15" else None)

    class _Res:
        def __init__(self, txt):
            self.stdout = txt.encode()

    def _fake_run(args, check, stdout):
        if args[1] == "-print-prog-name=llvm-config":
            return _Res("/usr/bin/llvm-config\n")
        return _Res("15.0.1\n")

    monkeypatch.setattr(extractors_pkg.subprocess, "run", _fake_run)
    with pytest.warns(UserWarning, match="found clang compiler"):
        got = extractors_pkg.clang_binary_path()
    assert got == "/usr/bin/clang-15"


def test_extractors_clang_binary_path_not_found(monkeypatch):
    extractors_pkg.clang_binary_path.cache_clear()
    monkeypatch.setattr(extractors_pkg, "_EXTENSION_AVAILABLE", True)
    monkeypatch.setattr(extractors_pkg, "LLVM_VERSION", "15.0.0")
    monkeypatch.setattr(extractors_pkg.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="cannot find clang compiler"):
        extractors_pkg.clang_binary_path()


def test_clang_driver_scoped_options_restores_state(monkeypatch):
    class _DummyIncludeType:
        User = "user"

    class _DummyClangDriverType:
        IncludeDirType = _DummyIncludeType

    class _DummyDriver:
        def __init__(self):
            self.filename = "old.c"
            self.added = []
            self.removed = []

        def getFileName(self):
            return self.filename

        def setFileName(self, name):
            self.filename = name

        def addIncludeDir(self, path, typ):
            self.added.append((path, typ))

        def removeIncludeDir(self, path, typ):
            self.removed.append((path, typ))

    monkeypatch.setattr(extractors_pkg, "ClangDriver", _DummyClangDriverType)
    d = _DummyDriver()
    with extractors_pkg.clang_driver_scoped_options(d, additional_include_dir="/inc", filename="new.c") as scoped:
        assert scoped is d
        assert d.filename == "new.c"
        assert d.added == [("/inc", "user")]
    assert d.filename == "old.c"
    assert d.removed == [("/inc", "user")]

