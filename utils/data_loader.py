# utils/data_loader.py
import pandas as pd
import torch
from torch_geometric.data import Data

def load_transaction_df(df_or_path):
    """
    Accept either a pandas.DataFrame or a path to CSV.
    Returns: Data (PyG), user_ids (list mapping node->original id or name),
             src_list (list of source node ids), dst_list, amounts (list)
    Expected CSV columns (auto-detected): sender/receiver OR nameOrig/nameDest OR source/target
    Also looks for 'amount' or 'amount_transferred' or 'step' etc. Labels look for 'isFraud' or 'label'.
    """
    if isinstance(df_or_path, str):
        df = pd.read_csv(df_or_path)
    else:
        df = df_or_path.copy()

    # detect sender/receiver
    if 'sender' in df.columns and 'receiver' in df.columns:
        src_col, dst_col = 'sender', 'receiver'
    elif 'nameOrig' in df.columns and 'nameDest' in df.columns:
        src_col, dst_col = 'nameOrig', 'nameDest'
    elif 'source' in df.columns and 'target' in df.columns:
        src_col, dst_col = 'source', 'target'
    else:
        # try common possibilities
        possible = [c.lower() for c in df.columns]
        raise ValueError("CSV must contain sender/receiver or nameOrig/nameDest or source/target columns. Found: " + ", ".join(df.columns))

    # amount column detection
    amount_col = None
    for cand in ['amount', 'amt', 'transaction_amt', 'amount_transferred']:
        if cand in df.columns:
            amount_col = cand
            break

    # label detection
    label_col = None
    for cand in ['isFraud', 'label', 'fraud', 'is_fraud']:
        if cand in df.columns:
            label_col = cand
            break

    # Map unique user ids to integer node ids
    all_users = pd.concat([df[src_col].astype(str), df[dst_col].astype(str)], axis=0).unique().tolist()
    user2node = {u: i for i, u in enumerate(all_users)}
    node2user = {i: u for u, i in user2node.items()}

    # Build edge index arrays in the same order as dataframe rows
    src_nodes = [user2node[str(x)] for x in df[src_col].tolist()]
    dst_nodes = [user2node[str(x)] for x in df[dst_col].tolist()]

    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)

    # Basic node features: identity (one-hot) can be huge; we use simple degree features:
    num_nodes = len(all_users)
    deg = [0] * num_nodes
    for s in src_nodes:
        deg[s] += 1
    for d in dst_nodes:
        deg[d] += 1
    # feature: [degree, 1] as float
    x = torch.tensor([[float(d), 1.0] for d in deg], dtype=torch.float)

    # Labels per edge (transaction-level)
    if label_col:
        y = torch.tensor(df[label_col].astype(int).tolist(), dtype=torch.long)
    else:
        y = None

    # Amounts
    if amount_col:
        amounts = df[amount_col].tolist()
    else:
        amounts = [None] * len(src_nodes)

    data = Data(x=x, edge_index=edge_index, y=y)
    data.num_nodes = num_nodes
    data.user_map = node2user  # map node id -> original user id string
    data.src_nodes = src_nodes
    data.dst_nodes = dst_nodes
    data.amounts = amounts

    return data, node2user, src_nodes, dst_nodes, amounts
