import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class MultiTaskGNN(nn.Module):
    def __init__(self, in_channels, edge_feat_dim=1, hidden_dim=64):
        super(MultiTaskGNN, self).__init__()

        # GNN for node embeddings
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Edge-level fraud classifier (binary)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Node-level suspiciousness classifier (binary)
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x, edge_index, edge_attr):
        # GNN node embeddings
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))

        # Predict edge-level fraud
        src, dst = edge_index
        edge_input = torch.cat([x[src], x[dst], edge_attr], dim=1)
        edge_out = self.edge_mlp(edge_input).squeeze()

        # Predict node-level suspiciousness
        node_out = self.node_mlp(x).squeeze()

        return edge_out, node_out, x
