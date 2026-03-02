"""
AST node wrappers for TrVD sub-tree decomposition.

Ported from TrVD (https://github.com/XUPT-SSS/TrVD) ``tree.py``.
"""

from __future__ import annotations

import tree_sitter


def _needs_splitting(node, max_depth: int = 8, max_size: int = 40) -> bool:
    """Return True if the sub-tree rooted at *node* exceeds thresholds."""
    return _get_max_depth(node) > max_depth or _get_tree_size(node) > max_size


def _get_max_depth(node) -> int:
    if not node:
        return 0
    if len(node.children) == 0:
        return 1
    return 1 + max(_get_max_depth(c) for c in node.children)


def _get_tree_size(node) -> int:
    if not node:
        return 0
    if len(node.children) == 0:
        return 1
    return 1 + sum(_get_tree_size(c) for c in node.children)


class ASTNode:
    """Wrapper around a tree-sitter node that optionally truncates children
    at compound-statement boundaries when *do_split* is True."""

    _BLOCK_TYPES = frozenset([
        'function_definition', 'if_statement', 'try_statement',
        'for_statement', 'switch_statement', 'while_statement',
        'do_statement', 'catch_clause', 'case_statement',
    ])

    def __init__(self, node, do_split: bool = True):
        self.node = node
        self.do_split = do_split
        self.is_leaf = self._is_leaf_node()
        self.token = self._get_token()
        self.children: list[ASTNode] = self._add_children()

    def _is_leaf_node(self) -> bool:
        if isinstance(self.node, tree_sitter.Tree):
            return len(self.node.root_node.children) == 0
        return len(self.node.children) == 0

    def _get_token(self):
        if isinstance(self.node, tree_sitter.Tree):
            n = self.node.root_node
        else:
            n = self.node
        if self.is_leaf:
            return n.text
        return n.type

    def _add_children(self) -> list[ASTNode]:
        if self.is_leaf:
            return []
        children = self.node.children
        if not self.do_split:
            return [ASTNode(c, self.do_split) for c in children]
        if self.token in self._BLOCK_TYPES:
            body_idx = 0
            for c in children:
                if c.type == 'compound_statement':
                    break
                body_idx += 1
            return [ASTNode(children[i], self.do_split) for i in range(body_idx)]
        return [ASTNode(c, self.do_split) for c in children]


class SingleNode:
    """Lightweight node that only stores its own token (no recursive children)."""

    def __init__(self, node):
        self.node = node
        self.is_leaf = self._is_leaf_node()
        self.token = self._get_token()
        self.children: list = []

    def _is_leaf_node(self) -> bool:
        if isinstance(self.node, tree_sitter.Tree):
            return len(self.node.root_node.children) == 0
        return len(self.node.children) == 0

    def _get_token(self):
        if isinstance(self.node, tree_sitter.Tree):
            n = self.node.root_node
        else:
            n = self.node
        if self.is_leaf:
            return n.text
        return n.type
