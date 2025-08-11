import torch
import torch.nn as nn

class GroupClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=2):
        super(GroupClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
