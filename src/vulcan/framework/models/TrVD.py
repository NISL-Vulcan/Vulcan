"""
TrVD: Tree-decomposition based Vulnerability Detector.

Adapted from https://github.com/XUPT-SSS/TrVD for the Vulcan framework.

Architecture:
  1. BatchTreeEncoder -- recursive tree-structured NN with GRU + attention
     that encodes each AST sub-tree into a fixed-size vector.
  2. Transformer encoder -- summarises the sequence of sub-tree embeddings
     via self-attention, followed by max-pooling and linear classification.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def _matrix_mul(input_tensor, weight, bias=None):
    """Batched matrix multiply with optional bias and tanh activation."""
    features = []
    for feature in input_tensor:
        feature = torch.mm(feature, weight)
        if isinstance(bias, nn.Parameter):
            feature = feature + bias.expand(feature.size(0), bias.size(1))
        feature = torch.tanh(feature).unsqueeze(0)
        features.append(feature)
    return torch.cat(features, 0).squeeze(-1)


def _element_wise_mul(input1, input2):
    """Element-wise weighted sum across the first (child) dimension."""
    features = []
    for f1, f2 in zip(input1, input2):
        f2 = f2.unsqueeze(1).expand_as(f1)
        features.append((f1 * f2).unsqueeze(0))
    output = torch.cat(features, 0)
    return torch.sum(output, 0).unsqueeze(0)


class BatchTreeEncoder(nn.Module):
    """Recursively encodes batched sub-trees using GRU aggregation with
    child-attention."""

    def __init__(self, vocab_size: int, embedding_dim: int, encode_dim: int,
                 device, pretrained_weight=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encode_dim = encode_dim
        self.W_c = nn.Linear(embedding_dim, encode_dim)
        self.activation = F.relu
        self.device = device
        self.node_list = []
        self.batch_node = None

        if pretrained_weight is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))

        self.agg_net = nn.GRU(embedding_dim, encode_dim, 1)
        self.W_r = nn.Linear(encode_dim, encode_dim)

        self.sent_weight = nn.Parameter(torch.Tensor(encode_dim, encode_dim))
        self.sent_bias = nn.Parameter(torch.Tensor(1, encode_dim))
        self.context_weight = nn.Parameter(torch.Tensor(encode_dim, 1))
        self._init_weights()

    def _init_weights(self, mean: float = 0.0, std: float = 0.05):
        self.sent_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)
        self.sent_bias.data.normal_(mean, std)

    def _create_tensor(self, tensor):
        return tensor.to(self.device)

    def _traverse_mul(self, node, batch_index):
        size = len(node)
        if not size:
            return None

        batch_current = self._create_tensor(Variable(torch.zeros(size, self.encode_dim)))
        index, children_index = [], []
        current_node, children = [], []

        for i in range(size):
            if node[i][0] != -1:
                index.append(i)
                current_node.append(node[i][0])
                temp = node[i][1:]
                for j in range(len(temp)):
                    if temp[j][0] != -1:
                        if len(children_index) <= j:
                            children_index.append([i])
                            children.append([temp[j]])
                        else:
                            children_index[j].append(i)
                            children[j].append(temp[j])
            else:
                batch_index[i] = -1

        batch_current = batch_current.index_copy(
            0,
            Variable(torch.LongTensor(index).to(self.device)),
            self.embedding(Variable(torch.LongTensor(current_node).to(self.device))),
        )

        childs_hidden_sum = self._create_tensor(Variable(torch.zeros(size, self.encode_dim)))
        hidden_per_child = []

        for c in range(len(children)):
            zeros = self._create_tensor(Variable(torch.zeros(size, self.encode_dim)))
            batch_children_index = [batch_index[i] for i in children_index[c]]
            tree = self._traverse_mul(children[c], batch_children_index)
            if tree is not None:
                cur_child_hidden = zeros.index_copy(
                    0,
                    Variable(torch.LongTensor(children_index[c]).to(self.device)),
                    tree,
                )
                childs_hidden_sum += cur_child_hidden
                hidden_per_child.append(cur_child_hidden)

        if hidden_per_child:
            child_hiddens = torch.stack(hidden_per_child)
            childs_weighted = _matrix_mul(child_hiddens, self.sent_weight, self.sent_bias)
            childs_weighted = _matrix_mul(childs_weighted, self.context_weight).permute(1, 0)
            childs_weighted = F.softmax(childs_weighted, dim=-1)
            childs_hidden_sum = _element_wise_mul(
                child_hiddens, childs_weighted.permute(1, 0)
            ).squeeze(0)

        batch_current = batch_current.unsqueeze(0)
        childs_hidden_sum = childs_hidden_sum.unsqueeze(0)
        _, hn = self.agg_net(batch_current, childs_hidden_sum)
        hn = hn.squeeze(0)

        batch_index = [i for i in batch_index if i != -1]
        b_in = Variable(torch.LongTensor(batch_index).to(self.device))
        nd_tmp = self.batch_node.index_copy(0, b_in, hn)
        self.node_list.append(nd_tmp)

        return hn

    def forward(self, x, bs: int):
        self.batch_node = self._create_tensor(Variable(torch.zeros(bs, self.encode_dim)))
        self.node_list = []
        self._traverse_mul(x, list(range(bs)))
        self.node_list = torch.stack(self.node_list)
        return torch.max(self.node_list, 0)[0]


class TrVD(nn.Module):
    """TrVD vulnerability detector.

    Constructor parameters (passed from ``MODEL.PARAMS`` in YAML config):
        embedding_dim (int): token embedding dimension (default 128)
        hidden_dim (int): BiLSTM hidden dimension (default 100)
        vocab_size (int): vocabulary size (+1 for unknown)
        encode_dim (int): sub-tree encoding dimension (default 128)
        label_size (int): number of output classes (default 2)
        n_heads (int): Transformer attention heads (default 4)
        n_transformer_layers (int): Transformer encoder layers (default 2)
        dropout (float): dropout rate (default 0.2)
        pretrained_weight_path (str | None): path to numpy array with
            pretrained Word2Vec embeddings
    """

    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 100,
                 vocab_size: int = 5000, encode_dim: int = 128,
                 label_size: int = 2, n_heads: int = 4,
                 n_transformer_layers: int = 2, dropout: float = 0.2,
                 pretrained_weight_path: str | None = None, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = 1
        self.embedding_dim = embedding_dim
        self.encode_dim = encode_dim
        self.label_size = label_size

        pretrained_weight = None
        if pretrained_weight_path:
            pretrained_weight = np.load(pretrained_weight_path)

        self.encoder = BatchTreeEncoder(
            vocab_size, embedding_dim, encode_dim,
            device=torch.device('cpu'),
            pretrained_weight=pretrained_weight,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=encode_dim, nhead=n_heads, batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_transformer_layers,
        )
        self.classifier = nn.Linear(encode_dim, label_size)
        self.dropout = nn.Dropout(dropout)

    def _get_zeros(self, num: int):
        zeros = torch.zeros(num, self.encode_dim)
        return zeros.to(next(self.parameters()).device)

    def _update_device(self):
        """Ensure sub-modules know the current device."""
        device = next(self.parameters()).device
        self.encoder.device = device

    def forward(self, x):
        """
        Args:
            x: list of samples, each sample is a list of sub-trees,
               each sub-tree is a nested list of token indices.
        Returns:
            logits: Tensor of shape ``(batch_size, label_size)``.
        """
        self._update_device()

        # filter out empty sub-trees
        filtered = []
        for sample in x:
            filtered.append([st for st in sample if len(st) > 1])

        batch_size = len(filtered)
        lens = [len(item) for item in filtered]
        max_len = max(lens) if lens else 1

        # flatten all sub-trees across the batch for the tree encoder
        all_subtrees = []
        for i in range(batch_size):
            for j in range(lens[i]):
                all_subtrees.append(filtered[i][j])

        if not all_subtrees:
            return self._get_zeros(batch_size).unsqueeze(1).expand(
                batch_size, self.label_size
            )

        encodes = self.encoder(all_subtrees, sum(lens))

        # re-assemble per-sample sequences, padding shorter ones
        seq, start = [], 0
        for i in range(batch_size):
            end = start + lens[i]
            if max_len - lens[i]:
                seq.append(self._get_zeros(max_len - lens[i]))
            seq.append(encodes[start:end])
            start = end

        encodes = torch.cat(seq)
        encodes = encodes.view(batch_size, max_len, -1)

        out = self.transformer_encoder(encodes)
        out = torch.transpose(out, 1, 2)
        out = F.max_pool1d(out, out.size(2)).squeeze(2)
        out = self.dropout(out)
        logits = self.classifier(out)
        return logits
