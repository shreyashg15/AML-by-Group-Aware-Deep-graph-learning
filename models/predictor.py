import pandas as pd
import networkx as nx

def predict_groups(transactions: pd.DataFrame, threshold: float = 0.5):
    """
    Predict groups and identify fraud nodes from transaction data.
    Fraud detection is based on the 'label' column.

    Args:
        transactions (pd.DataFrame): CSV data with at least ['source', 'target', 'label'] columns.
        threshold (float): Threshold above which a node is considered fraud if labels are probabilities.

    Returns:
        groups (list of sets): List of connected component groups.
        fraud_nodes (list): List of fraud node IDs.
    """
    if not {"source", "target", "label"}.issubset(transactions.columns):
        raise ValueError("CSV must contain 'source', 'target', and 'label' columns.")

    # Create directed graph
    G = nx.from_pandas_edgelist(transactions, source="source", target="target", create_using=nx.DiGraph())

    # Identify fraud nodes
    fraud_nodes = []
    for _, row in transactions.iterrows():
        label = row["label"]
        if isinstance(label, (int, float)):
            if label >= threshold:
                fraud_nodes.append(row["source"])
                fraud_nodes.append(row["target"])
        else:
            raise ValueError("Label column must contain numeric values (0/1 or probabilities).")

    fraud_nodes = list(set(fraud_nodes))

    # Grouping
    groups = [set(comp) for comp in nx.strongly_connected_components(G)]

    return groups, fraud_nodes
