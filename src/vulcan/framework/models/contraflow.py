import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool

class ASTNodeEncoder(nn.Module):
    def __init__(self, num_node_features, hidden_dim):
        super(ASTNodeEncoder, self).__init__()
        # Multi-layer GCN
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        # Two-layer graph convolution
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        return x

class StatementEncoder(nn.Module):
    def __init__(self, num_node_features, hidden_dim, output_dim):
        super(StatementEncoder, self).__init__()
        self.ast_node_encoder = ASTNodeEncoder(num_node_features, hidden_dim)
        self.fc = nn.Linear(2 * hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, node_features, edge_index):
        # AST node encoding
        node_embeddings = self.ast_node_encoder(node_features, edge_index)
        # Aggregate information from all nodes
        pooled = torch.cat([global_mean_pool(node_embeddings, torch.zeros(node_embeddings.size(0), dtype=torch.long, device=node_embeddings.device)),
                            global_max_pool(node_embeddings, torch.zeros(node_embeddings.size(0), dtype=torch.long, device=node_embeddings.device))], dim=1)
        pooled = self.dropout(pooled)
        return self.fc(pooled)

class ValueFlowPathEncoder(nn.Module):
    def __init__(self, num_node_features, hidden_dim, output_dim):
        super(ValueFlowPathEncoder, self).__init__()
        self.statement_encoder = StatementEncoder(num_node_features, hidden_dim, output_dim)
        self.bgru = nn.GRU(output_dim, hidden_dim, bidirectional=True)
        self.attention = nn.Linear(2 * hidden_dim, 1)

    def forward(self, x, edge_indices, lengths):
        # Process statements one by one
        encoded_statements = [self.statement_encoder(nodes, edges) for nodes, edges in zip(x, edge_indices)]
        # Sequence modeling
        encoded_sequence = torch.stack(encoded_statements)
        packed_sequence = pack_padded_sequence(encoded_sequence, lengths, enforce_sorted=False)
        output, _ = self.bgru(packed_sequence)
        output, _ = pad_packed_sequence(output)
        # Attention mechanism
        attention_weights = torch.softmax(self.attention(output), dim=0)
        output = torch.sum(attention_weights * output, dim=0)
        return output

class ContrastiveLearningModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ContrastiveLearningModel, self).__init__()
        self.vpe = ValueFlowPathEncoder(input_dim, hidden_dim, output_dim)

    def forward(self, path1, path2):
        rep1 = self.vpe(*path1)
        rep2 = self.vpe(*path2)
        return rep1, rep2

    def compute_contrastive_loss(self, rep1, rep2, temperature=0.5):
        # Compute cosine similarity
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        positive_similarity = cos(rep1, rep2)
        # Noise Contrastive Estimation (NCE) loss
        # This is a simplified version; real use may require harder negative mining.
        negative_similarity = cos(rep1, -rep2)
        loss = -torch.log(torch.exp(positive_similarity / temperature) / 
                          (torch.exp(positive_similarity / temperature) + torch.exp(negative_similarity / temperature)))
        return loss.mean()
