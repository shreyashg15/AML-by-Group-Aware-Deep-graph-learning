from utils.graph_utils import build_graph_from_csv
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 2)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        return self.conv2(x, edge_index)

data = build_graph_from_csv("data/converted.csv")
model = GCN(data.num_node_features, 16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(15):
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.y.long()], data.y.long())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"[Node Epoch {epoch+1}] Loss: {loss:.4f}")
