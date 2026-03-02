"""
AST decomposition utilities for TrVD.

Ported from TrVD (https://github.com/XUPT-SSS/TrVD) ``prepare_data.py``.
Provides functions to:
  - traverse a tree-sitter AST and collect token sequences
  - collect root-to-leaf paths for Word2Vec training
  - decompose an AST into ordered sub-trees of bounded size/depth
"""

from __future__ import annotations

import copy
from typing import List

import tree_sitter

from .tree_nodes import ASTNode, SingleNode, _needs_splitting


def get_sequences(node, sequence: list) -> None:
    """Pre-order traversal collecting token names into *sequence*."""
    current = SingleNode(node)

    if isinstance(node, tree_sitter.Tree):
        name = node.root_node.type
        children = node.root_node.children
    else:
        name = node.type
        children = node.children

    if name == 'comment':
        return

    sequence.append(current.token)

    for child in children:
        get_sequences(child, sequence)

    if isinstance(current.token, str) and current.token.lower() == 'compound_statement':
        sequence.append('End')


def get_root_paths(node, sequences: list, cur_path: list) -> None:
    """Collect all root-to-leaf paths for Word2Vec corpus construction."""
    current = SingleNode(node)

    if isinstance(node, tree_sitter.Tree):
        name = node.root_node.type
        children = node.root_node.children
    else:
        name = node.type
        children = node.children

    if current.is_leaf:
        if name == 'comment':
            return
        root_path = copy.deepcopy(cur_path)
        root_path.append(current.token)
        sequences.append(root_path)
        return

    if name == 'comment':
        return

    cur_path.append(current.token)
    for child in children:
        get_root_paths(child, sequences, copy.deepcopy(cur_path))


_BLOCK_STATEMENTS = frozenset([
    'function_definition', 'if_statement', 'try_statement',
    'for_statement', 'switch_statement', 'while_statement',
    'do_statement', 'catch_clause', 'case_statement',
])

_RECURSIVE_BLOCKS = frozenset([
    'if_statement', 'try_statement', 'for_statement',
    'switch_statement', 'while_statement', 'do_statement',
    'catch_clause',
])


def get_blocks(node, block_seq: list) -> None:
    """Decompose an AST into ordered sub-tree blocks stored in *block_seq*."""
    if isinstance(node, list):
        return

    if isinstance(node, tree_sitter.Tree):
        children = node.root_node.children
        name = node.root_node.type
    else:
        children = node.children
        name = node.type

    if name == 'comment':
        return

    if name in _BLOCK_STATEMENTS:
        do_split = _needs_splitting(node)
        if not do_split:
            block_seq.append(ASTNode(node, False))
            return

        block_seq.append(ASTNode(node, True))
        body_idx = 0
        for child in children:
            if child.type in ('compound_statement', 'expression_statement'):
                break
            body_idx += 1

        for i in range(body_idx, len(children)):
            child = children[i]
            if child.type == 'comment':
                continue
            if child.type not in _RECURSIVE_BLOCKS:
                block_seq.append(ASTNode(child, _needs_splitting(child)))
            get_blocks(child, block_seq)

    elif name == 'compound_statement':
        if isinstance(node, tree_sitter.Tree):
            cs_children = node.root_node.children
        else:
            cs_children = node.children

        for child in cs_children:
            if child.type == 'comment':
                continue
            if child.type not in _RECURSIVE_BLOCKS:
                block_seq.append(ASTNode(child, _needs_splitting(child)))
            else:
                get_blocks(child, block_seq)
    else:
        if isinstance(node, tree_sitter.Tree):
            for child in node.root_node.children:
                get_blocks(child, block_seq)
        else:
            for child in node.children:
                get_blocks(child, block_seq)
