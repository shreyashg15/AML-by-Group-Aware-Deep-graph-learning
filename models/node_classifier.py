import torch
from torch_geometric.nn import GCNConv

class NodeClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=32):
        super(NodeClassifier, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x.view(-1)

    def get_embeddings(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return torch.relu(x)
