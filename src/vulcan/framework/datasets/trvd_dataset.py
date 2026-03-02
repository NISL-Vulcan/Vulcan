"""
TrVD dataset: AST decomposition based vulnerability detection dataset.

Handles the full preprocessing pipeline:
  1. Load pickle data (code, label)
  2. Parse source code into ASTs via tree-sitter (C/C++)
  3. Train Word2Vec on AST token sequences and root-to-leaf paths
  4. Decompose each AST into bounded sub-trees
  5. Convert sub-tree tokens to vocabulary indices

Compatible with the Vulcan ``get_dataset`` / ``get_dataloader`` interface.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .trvd_utils.ast_decomposition import get_blocks, get_root_paths, get_sequences
from .trvd_utils.tree_nodes import ASTNode

# Vulcan already ships clean_gadget; reuse it.
from .vddata_utils.clean_gadget import clean_gadget


def _build_language_lib(grammar_dir: str, output_path: str):
    """Build the tree-sitter shared library on-the-fly if not present."""
    from tree_sitter import Language

    lang_dirs = []
    for name in ('tree-sitter-c', 'tree-sitter-cpp'):
        d = os.path.join(grammar_dir, name)
        if os.path.isdir(d):
            lang_dirs.append(d)
    if not lang_dirs:
        raise FileNotFoundError(
            f"No tree-sitter grammar directories found under {grammar_dir}. "
            "Expected tree-sitter-c and/or tree-sitter-cpp."
        )
    Language.build_library(output_path, lang_dirs)


def _get_parser(language_lib_path: str, language: str = 'cpp'):
    """Return a tree-sitter ``Parser`` configured for the given language."""
    from tree_sitter import Language, Parser

    lang = Language(language_lib_path, language)
    parser = Parser()
    parser.set_language(lang)
    return parser


def _normalize_code(code: str) -> str:
    """Strip comments and normalise identifiers using clean_gadget."""
    import re

    lines = code.split('\n')
    flat = ''
    for line in lines:
        line = line.strip()
        line = re.sub('//.*', '', line)
        flat += line + ' '
    flat = re.sub(r'/\*.*?\*/', '', flat)
    cleaned = clean_gadget([flat])
    return cleaned[0]


def _parse_ast(source: str, parser):
    """Parse source code string into a tree-sitter Tree."""
    try:
        encoded = source.encode('utf-8').decode('unicode_escape').encode()
    except (UnicodeDecodeError, UnicodeEncodeError):
        encoded = source.encode('utf-8')
    return parser.parse(encoded)


class TrVDDataset(Dataset):
    """Vulcan-compatible dataset for TrVD.

    Constructor kwargs (from ``DATASET.PARAMS``):
        dataset_path (str):  directory containing ``train.pkl``, ``val.pkl``,
            ``test.pkl`` (each with columns ``code`` and ``label``).
        language_lib (str):  path to compiled tree-sitter ``.so``.
            If empty, will try ``<dataset_path>/../../build_languages/my-languages.so``
            then fall back to building from grammars found under a ``languages/``
            sibling directory.
        grammar_dir (str):  directory with ``tree-sitter-c`` and ``tree-sitter-cpp``
            sub-directories (only needed when *language_lib* does not exist).
        w2v_path (str):  path to a pre-trained gensim Word2Vec model.
            If the file does not exist it will be trained from the training split.
        embedding_size (int):  Word2Vec embedding dimension (default 128).
        language (str):  tree-sitter language name (default ``'cpp'``).
        normalize (bool):  whether to apply code normalisation (default True).
        label_size (int):  number of classes (default 2 for binary).
    """

    def __init__(self, split: str, root: str, preprocess_format=None, **kwargs):
        super().__init__()
        args = kwargs.get('args', kwargs)

        dataset_path: str = args.get('dataset_path', root)
        self.embedding_size: int = args.get('embedding_size', 128)
        language: str = args.get('language', 'cpp')
        normalize: bool = args.get('normalize', True)
        self.n_classes: int = args.get('label_size', 2)

        language_lib: str = args.get('language_lib', '')
        grammar_dir: str = args.get('grammar_dir', '')
        w2v_path: str = args.get('w2v_path', '')

        # resolve split filename
        split_map = {'train': 'train.pkl', 'val': 'val.pkl', 'test': 'test.pkl'}
        pkl_file = os.path.join(dataset_path, split_map.get(split, 'val.pkl'))
        if not os.path.exists(pkl_file):
            raise FileNotFoundError(f"TrVD dataset split not found: {pkl_file}")

        # resolve language library
        if not language_lib:
            candidate = os.path.join(dataset_path, '..', '..', 'build_languages', 'my-languages.so')
            if os.path.isfile(candidate):
                language_lib = candidate
        if not language_lib or not os.path.isfile(language_lib):
            lib_out = os.path.join(dataset_path, 'my-languages.so')
            if not grammar_dir:
                grammar_dir = os.path.join(dataset_path, '..', '..', 'languages')
            if os.path.isdir(grammar_dir):
                _build_language_lib(grammar_dir, lib_out)
                language_lib = lib_out
            else:
                raise FileNotFoundError(
                    f"Cannot find tree-sitter grammar directory at {grammar_dir} "
                    "and no pre-built language_lib specified."
                )

        parser = _get_parser(language_lib, language)

        # decide w2v path
        subtree_dir = os.path.join(dataset_path, 'subtrees')
        os.makedirs(subtree_dir, exist_ok=True)
        if not w2v_path:
            w2v_path = os.path.join(subtree_dir, f'node_w2v_{self.embedding_size}')

        # try to load pre-processed block pickle first
        block_pkl = os.path.join(subtree_dir, f'{split}_block.pkl')
        if os.path.exists(block_pkl) and os.path.exists(w2v_path):
            data = pd.read_pickle(block_pkl)
            data = data.drop(data[data['code'].str.len() == 0].index)
            self._data = data
            self._load_vocab(w2v_path)
            return

        # load raw data
        raw = pd.read_pickle(pkl_file)

        # normalise
        if normalize:
            raw['code'] = raw['code'].apply(_normalize_code)

        # parse ASTs
        raw['code'] = raw['code'].apply(lambda s: _parse_ast(s, parser))

        # train w2v on the training corpus (only when processing the train split
        # or when w2v model does not yet exist)
        if not os.path.exists(w2v_path):
            train_raw = raw if split == 'train' else pd.read_pickle(
                os.path.join(dataset_path, 'train.pkl')
            )
            if normalize and split != 'train':
                train_raw['code'] = train_raw['code'].apply(_normalize_code)
                train_raw['code'] = train_raw['code'].apply(lambda s: _parse_ast(s, parser))
            self._train_w2v(train_raw, w2v_path)

        self._load_vocab(w2v_path)

        # decompose and index
        keep = copy.deepcopy(raw)
        keep['code'] = keep['code'].apply(self._ast_to_indexed_blocks)
        keep.to_pickle(block_pkl)
        keep = keep.drop(keep[keep['code'].str.len() == 0].index)
        self._data = keep

    # ------------------------------------------------------------------
    # Word2Vec helpers
    # ------------------------------------------------------------------
    def _train_w2v(self, df: pd.DataFrame, save_path: str):
        from gensim.models.word2vec import Word2Vec

        def _to_corpus(ast):
            seq = []
            get_sequences(ast, seq)
            paths = []
            get_root_paths(ast, paths, [])
            paths.append(seq)
            return paths

        corpus_nested = df['code'].apply(_to_corpus)
        corpus = []
        for paths in corpus_nested:
            for path in paths:
                path = [t.decode('utf-8') if isinstance(t, bytes) else str(t) for t in path]
                corpus.append(path)

        w2v = Word2Vec(corpus, vector_size=self.embedding_size, workers=4, sg=1, min_count=3)
        w2v.save(save_path)

    def _load_vocab(self, w2v_path: str):
        from gensim.models.word2vec import Word2Vec

        wv = Word2Vec.load(w2v_path).wv
        self._vocab = wv.key_to_index
        self._max_token = wv.vectors.shape[0]
        self.vocab_size = self._max_token + 1
        self.pretrained_embeddings = np.zeros(
            (self.vocab_size, wv.vectors.shape[1]), dtype='float32',
        )
        self.pretrained_embeddings[:self._max_token] = wv.vectors

    def _ast_to_indexed_blocks(self, ast_tree) -> list:
        """Decompose AST tree into blocks, convert tokens to indices."""
        blocks = []
        get_blocks(ast_tree, blocks)
        result = []
        for b in blocks:
            result.append(self._tree_to_index(b))
        return result

    def _tree_to_index(self, node) -> list:
        token = node.token
        if isinstance(token, bytes):
            token = token.decode('utf-8')
        else:
            token = str(token)
        idx = self._vocab.get(token, self._max_token)
        result = [idx]
        for child in node.children:
            result.append(self._tree_to_index(child))
        return result

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        row = self._data.iloc[index]
        subtrees = row['code']
        label = torch.tensor(row['label'], dtype=torch.long)
        return subtrees, label
