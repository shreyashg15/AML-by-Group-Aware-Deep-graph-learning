import pandas as pd
import torch
from torch_geometric.data import Data

def build_graph_from_csv(csv_path):
    df = pd.read_csv(csv_path)

    # 🔍 Auto-detect column names for sender and receiver
    if 'sender' in df.columns and 'receiver' in df.columns:
        src_col, dst_col = 'sender', 'receiver'
    elif 'nameOrig' in df.columns and 'nameDest' in df.columns:
        src_col, dst_col = 'nameOrig', 'nameDest'
    elif 'source' in df.columns and 'target' in df.columns:
        src_col, dst_col = 'source', 'target'
    else:
        raise ValueError("❌ Unsupported CSV format: must include sender/receiver columns")

    # 🧠 Build user node list and mapping
    users = pd.concat([df[src_col], df[dst_col]]).unique()
    user2id = {user: idx for idx, user in enumerate(users)}

    df['src'] = df[src_col].map(user2id)
    df['dst'] = df[dst_col].map(user2id)

    # 🧩 Graph edges
    edge_index = torch.tensor([df['src'].values, df['dst'].values], dtype=torch.long)

    # 🧬 Node features — use simple identity matrix (one-hot)
    x = torch.eye(len(user2id), dtype=torch.float)

    # 🎯 Labels (transaction-level)
    if 'isFraud' in df.columns:
        y = torch.tensor(df['isFraud'].values, dtype=torch.long)
    elif 'label' in df.columns:
        y = torch.tensor(df['label'].values, dtype=torch.long)
    else:
        raise ValueError("❌ Missing label column ('isFraud' or 'label') in dataset")

    # 📦 Build PyG graph
    data = Data(x=x, edge_index=edge_index, y=y)

    return data
