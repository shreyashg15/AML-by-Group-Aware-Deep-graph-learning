# app_gagnn_v2.py
"""
Optimized GAGNN/GAE Streamlit app (v2) with:
 - Static graph styled like screenshot (gradient colors, directed edges, node size proportional to fraud prob)
 - Training progress bar + loss only in Streamlit
 - Total training time display
 - Sortable summary table for all nodes
 - Sorted high-risk node table
 - Mini-batch neighbor sampling for speed
 - Early stopping
 - GPU support
 - Custom save name + fixed save names + fingerprint save
Usage:
    streamlit run app_gagnn_v2.py
"""
import os
import hashlib
import time
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import networkx as nx
import streamlit as st
import plotly.graph_objects as go

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# -------------------------
# Utilities
# -------------------------
def ensure_dirs():
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(parents=True, exist_ok=True)

def dataset_fingerprint(df: pd.DataFrame) -> str:
    h = hashlib.sha1(pd.util.hash_pandas_object(df, index=True).values).hexdigest()
    return h[:16]


# -------------------------
# Build graph & node features
# -------------------------
def build_graph_from_df(df: pd.DataFrame, src_col="source", tgt_col="target", amount_col="amount") -> Tuple[Data, dict]:
    unique_nodes = pd.Index(df[src_col].astype(str).tolist() + df[tgt_col].astype(str).tolist()).unique()
    node_to_idx = {n: i for i, n in enumerate(unique_nodes)}
    idx_to_node = {i: n for n, i in node_to_idx.items()}

    src = df[src_col].astype(str).map(node_to_idx).to_numpy(dtype=np.int64)
    tgt = df[tgt_col].astype(str).map(node_to_idx).to_numpy(dtype=np.int64)
    edge_index = np.vstack([src, tgt])

    Gnx = nx.DiGraph()
    for s, t, row in zip(src, tgt, df.itertuples(index=False)):
        amt = float(getattr(row, amount_col)) if amount_col in df.columns else 0.0
        if Gnx.has_edge(s, t):
            Gnx[s][t]["amounts"].append(amt)
        else:
            Gnx.add_edge(s, t, amounts=[amt])

    n_nodes = len(unique_nodes)
    in_deg = np.zeros(n_nodes)
    out_deg = np.zeros(n_nodes)
    sum_amt = np.zeros(n_nodes)
    mean_amt = np.zeros(n_nodes)
    std_amt = np.zeros(n_nodes)
    max_amt = np.zeros(n_nodes)
    min_amt = np.zeros(n_nodes)

    for u in range(n_nodes):
        outs_amt = [a for _, _, d in Gnx.out_edges(u, data=True) for a in d["amounts"]]
        ins_amt = [a for _, _, d in Gnx.in_edges(u, data=True) for a in d["amounts"]]
        all_amt = np.array(outs_amt + ins_amt) if outs_amt or ins_amt else np.array([0.0])
        sum_amt[u] = all_amt.sum()
        mean_amt[u] = all_amt.mean()
        std_amt[u] = all_amt.std() if len(all_amt) > 1 else 0.0
        max_amt[u] = all_amt.max()
        min_amt[u] = all_amt.min()
        out_deg[u] = len(outs_amt)
        in_deg[u] = len(ins_amt)

    total_deg = in_deg + out_deg
    log_sum = np.log1p(sum_amt)
    ratio_io = (in_deg + 1) / (out_deg + 1)

    feats = np.vstack([in_deg, out_deg, total_deg,
                       sum_amt, mean_amt, std_amt,
                       max_amt, min_amt, log_sum, ratio_io]).T
    scaler = StandardScaler()
    feats = scaler.fit_transform(feats)

    x = torch.tensor(feats, dtype=torch.float)
    edge_index_t = torch.tensor(edge_index, dtype=torch.long)
    undirected = to_undirected(edge_index_t)

    data = Data(x=x, edge_index=undirected)
    data.num_nodes = n_nodes
    mapping = {"node_to_idx": node_to_idx, "idx_to_node": idx_to_node, "scaler": scaler}
    return data, mapping


# -------------------------
# Model
# -------------------------
class SageEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels=256, out_channels=128, dropout=0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x, edge_index):
        x = self.act(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.act(self.bn2(self.conv2(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv3(x, edge_index)


# -------------------------
# Training (mini-batch) with progress in Streamlit
# -------------------------
def load_or_train_optimized(df, save_name, epochs, lr, batch_size, patience, device_str):
    ensure_dirs()
    fingerprint = dataset_fingerprint(df)
    model_path = Path("models") / f"{fingerprint}_optimized_gagnn.pth"
    pred_path = Path("models") / f"{fingerprint}_node_predictions.csv"

    custom_model_path = Path("models") / f"{save_name}.pth" if save_name else None
    custom_pred_path = Path("models") / f"{save_name}_predictions.csv" if save_name else None

    fixed_model_path = Path("models") / "gagnn_model.pth"
    fixed_pred_path = Path("models") / "node_predictions.csv"

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    if model_path.exists() and pred_path.exists():
        preds = pd.read_csv(pred_path)
        st.success("Model loaded from saved file.")
        return preds, str(model_path)

    # Build graph
    data, mapping = build_graph_from_df(df)
    data = data.to(device)

    encoder = SageEncoder(data.x.size(1)).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    loader = NeighborLoader(data, num_neighbors=[15, 10], batch_size=batch_size)

    best_loss = float("inf")
    patience_count = 0
    start_time = time.time()

    progress_bar = st.progress(0)
    status_text = st.empty()

    for epoch in range(1, epochs + 1):
        encoder.train()
        total_loss = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            z = encoder(batch.x, batch.edge_index)
            pos_edge = batch.edge_index
            if pos_edge.size(1) == 0:
                continue
            pos_scores = (z[pos_edge[0]] * z[pos_edge[1]]).sum(dim=1)
            num_neg = pos_scores.size(0)
            neg_u = torch.randint(0, batch.num_nodes, (num_neg,), device=device)
            neg_v = torch.randint(0, batch.num_nodes, (num_neg,), device=device)
            neg_scores = (z[neg_u] * z[neg_v]).sum(dim=1)
            logits = torch.cat([pos_scores, neg_scores])
            labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        progress_bar.progress(epoch / epochs)
        status_text.text(f"Epoch {epoch}/{epochs} — Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_count = 0
            torch.save(encoder.state_dict(), model_path)
        else:
            patience_count += 1
            if patience_count >= patience:
                break

    elapsed = time.time() - start_time
    encoder.load_state_dict(torch.load(model_path, map_location=device))
    encoder.eval()

    # Full embeddings
    full_loader = NeighborLoader(data, num_neighbors=[15, 10], batch_size=batch_size)
    embeddings = []
    with torch.no_grad():
        for batch in full_loader:
            batch = batch.to(device)
            z = encoder(batch.x, batch.edge_index)
            embeddings.append(z[:batch.batch_size].cpu().numpy())
    embeddings = np.vstack(embeddings)

    kmeans = KMeans(n_clusters=min(50, max(2, int(np.sqrt(data.num_nodes)))), random_state=42)
    groups = kmeans.fit_predict(embeddings)
    dists = np.linalg.norm(embeddings - kmeans.cluster_centers_[groups], axis=1)
    fraud_prob = MinMaxScaler().fit_transform(dists.reshape(-1, 1)).flatten()

    preds_df = pd.DataFrame({
        "node_idx": range(data.num_nodes),
        "node": [mapping["idx_to_node"][i] for i in range(data.num_nodes)],
        "group": groups,
        "anomaly_score": dists,
        "fraud_prob": fraud_prob
    })

    preds_df.to_csv(pred_path, index=False)
    torch.save(encoder.state_dict(), model_path)
    if save_name:
        preds_df.to_csv(custom_pred_path, index=False)
        torch.save(encoder.state_dict(), custom_model_path)
    preds_df.to_csv(fixed_pred_path, index=False)
    torch.save(encoder.state_dict(), fixed_model_path)

    st.success(f"Training complete in {elapsed:.2f} seconds.")
    st.info(f"Saved model: {model_path}\nCustom: {custom_model_path}\nFixed: {fixed_model_path}")
    return preds_df, str(model_path)


# -------------------------
# Static Graph
# -------------------------
def plot_static_graph(G: nx.DiGraph, preds_df: pd.DataFrame):
    pos = nx.spring_layout(G, seed=42)
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='black'),
                            hoverinfo='none', mode='lines')

    node_x, node_y, node_color, node_size, node_text = [], [], [], [], []
    for n in G.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        prob = preds_df.set_index("node").loc[n, "fraud_prob"]
        node_color.append(prob)
        node_size.append(10 + 30 * prob)
        node_text.append(str(n))

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text', text=node_text,
        textposition="middle center",
        marker=dict(showscale=True, colorscale="RdYlBu_r", color=node_color,
                    size=node_size, colorbar=dict(title="Fraud Probability")),
        hoverinfo="text"
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(title="Static Graph (Gradient-based)", showlegend=False,
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      plot_bgcolor='white')
    return fig


# -------------------------
# Main App
# -------------------------
def main():
    st.set_page_config(layout="wide", page_title="GAGNN Optimized v2")
    st.title("GAGNN Optimized v2 — Static Graph + Sortable Tables")

    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    use_default = st.sidebar.checkbox("Use default file")
    save_name = st.sidebar.text_input("Custom Save Name (no extension)", value="")
    epochs = st.sidebar.number_input("Epochs", 10, 2000, 200)
    lr = st.sidebar.number_input("Learning Rate", 1e-5, 1e-1, 1e-3, format="%.6f")
    batch_size = st.sidebar.number_input("Batch Size", 128, 50000, 2048)
    patience = st.sidebar.number_input("Patience", 1, 200, 10)
    threshold = st.sidebar.slider("Suspicion Threshold", 0.0, 1.0, 0.5, 0.01)
    device_choice = st.sidebar.selectbox("Device", ["cuda", "cpu"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    elif use_default:
        if os.path.exists("data/converted01.csv"):
            df = pd.read_csv("data/converted01.csv")
        elif os.path.exists("data/transactions01.csv"):
            df = pd.read_csv("data/transactions01.csv")
        else:
            st.error("No default file found.")
            return
    else:
        st.warning("Upload a file or select default option.")
        return

    if "nameOrig" in df.columns and "nameDest" in df.columns:
        df = df.rename(columns={"nameOrig": "source", "nameDest": "target"})

    preds_df, _ = load_or_train_optimized(df, save_name, epochs, lr, batch_size, patience, device_choice)

    # Static graph
    G = nx.from_pandas_edgelist(df, source="source", target="target", edge_attr="amount", create_using=nx.DiGraph)
    st.plotly_chart(plot_static_graph(G, preds_df), use_container_width=True)

    # Summary table (sortable)
    st.subheader("Summary Table — All Nodes")
    st.dataframe(preds_df.sort_values("fraud_prob", ascending=False))

    # High-risk nodes
    st.subheader(f"High-Risk Nodes (Fraud Probability >= {threshold})")
    high_risk = preds_df[preds_df["fraud_prob"] >= threshold].sort_values("fraud_prob", ascending=False)
    st.dataframe(high_risk)
    # --- Add footer ---
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        right: 0;
        padding: 5px 10px;
        font-size: 12px;
        color: rgba(100, 100, 100, 0.7); /* Light grey color */
    }
    </style>
    <div class="footer">
        Project developed by ShreyasHG &amp; Team
    </div>
    """,
    unsafe_allow_html=True
)



if __name__ == "__main__":
    main()
