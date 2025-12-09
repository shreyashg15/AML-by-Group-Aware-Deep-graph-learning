# app_gagnn.py
# Unsupervised GAE + clustering (adapted) with improved Streamlit UX:
# - Progress bar + live loss in Streamlit only
# - Total training time and completion message with save paths
# - Custom save-name option
# - Sortable summary table and high-risk node table (sorted)
# - Added Evaluation Metrics (Accuracy, Precision, Recall, F1 Score, AUC-ROC)

import os
import hashlib
import json
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import networkx as nx
import streamlit as st
import plotly.graph_objects as go

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import to_undirected, negative_sampling, train_test_split_edges
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GAE

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# -------------------------
# (UNCHANGED) Visualization functions
# -------------------------
import matplotlib.pyplot as plt

def visualize_tree_graph_static(G, fraud_nodes):
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, seed=42)

    normal_nodes = [n for n in G.nodes() if n not in fraud_nodes]

    nx.draw_networkx_edges(G, pos, alpha=0.4)
    nx.draw_networkx_nodes(G, pos, nodelist=normal_nodes,
                           node_color="lightblue", node_size=300)
    nx.draw_networkx_nodes(G, pos, nodelist=list(fraud_nodes),
                           node_color="red", node_size=350)

    nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title("Static Fraud Tree Graph")
    plt.axis("off")
    return plt

def visualize_graph(G, fraud_nodes, pos, groups, mode="Static", transactions=None):
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_x, node_y, node_color, node_border_color, hover_text = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        if node in fraud_nodes:
            node_color.append('lightcoral')
            node_border_color.append('red')
        else:
            node_color.append('lightblue')
            node_border_color.append('black')
        hover_text.append(f"User ID: {node}")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[str(n) for n in G.nodes()],
        hovertext=hover_text,
        textposition="top center",
        marker=dict(
            color=node_color,
            size=20,
            line=dict(width=2, color=node_border_color)
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="Interactive Fraud Graph",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    return fig

def visualize_tree_graph_plotly(G, fraud_nodes):
    pos = nx.spring_layout(G, k=15, iterations=300, seed=42)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
    node_x, node_y, node_color, node_border_color, hover_text = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        if node in fraud_nodes:
            node_color.append('lightcoral')
            node_border_color.append('red')
        else:
            node_color.append('lightblue')
            node_border_color.append('black')
        hover_text.append(f"Node: {node}")
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[str(n) for n in G.nodes()],
        hoverinfo='text',
        hovertext=hover_text,
        textposition="top center",
        marker=dict(color=node_color, size=20, line=dict(width=2, color=node_border_color))
    )
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="Interactive Tree Graph - Money Laundering Flow",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    return fig

# -------------------------
# Data & model helpers
# -------------------------
def dataset_fingerprint(df: pd.DataFrame) -> str:
    h = hashlib.sha1(pd.util.hash_pandas_object(df, index=True).values).hexdigest()
    return h[:16]

def ensure_dirs():
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("models_path").mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(parents=True, exist_ok=True)

def build_graph_from_df(df: pd.DataFrame, src_col="source", tgt_col="target", amount_col="amount") -> Tuple[Data, dict]:
    if src_col not in df.columns or tgt_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{src_col}' and '{tgt_col}' columns")
    unique_nodes = pd.Index(df[src_col].astype(str).tolist() + df[tgt_col].astype(str).tolist()).unique()
    node_to_idx = {n: i for i, n in enumerate(unique_nodes)}
    idx_to_node = {i: n for n, i in node_to_idx.items()}

    edge_src = df[src_col].astype(str).map(node_to_idx).to_numpy(dtype=np.int64)
    edge_tgt = df[tgt_col].astype(str).map(node_to_idx).to_numpy(dtype=np.int64)
    edge_index = np.vstack([edge_src, edge_tgt])

    Gnx = nx.DiGraph()
    for s, t, row in zip(edge_src, edge_tgt, df.itertuples(index=False)):
        amt = float(getattr(row, amount_col)) if amount_col in df.columns else 0.0
        if Gnx.has_edge(s, t):
            Gnx[s][t]["amounts"].append(amt)
        else:
            Gnx.add_edge(s, t, amounts=[amt])

    num_nodes = len(unique_nodes)
    in_deg = np.zeros(num_nodes, dtype=float)
    out_deg = np.zeros(num_nodes, dtype=float)
    sum_amount = np.zeros(num_nodes, dtype=float)
    mean_amount = np.zeros(num_nodes, dtype=float)
    std_amount = np.zeros(num_nodes, dtype=float)
    max_amount = np.zeros(num_nodes, dtype=float)
    min_amount = np.zeros(num_nodes, dtype=float)

    for u in range(num_nodes):
        outs = list(Gnx.out_edges(u, data=True))
        outs_amounts = []
        for _u, v, data in outs:
            outs_amounts.extend(data.get("amounts", []))
        ins = list(Gnx.in_edges(u, data=True))
        ins_amounts = []
        for _v, _u2, data in ins:
            ins_amounts.extend(data.get("amounts", []))
        all_amounts = np.array(outs_amounts + ins_amounts, dtype=float) if (outs_amounts + ins_amounts) else np.array([0.0])
        sum_amount[u] = all_amounts.sum()
        mean_amount[u] = all_amounts.mean() if len(all_amounts) > 0 else 0.0
        std_amount[u] = all_amounts.std() if len(all_amounts) > 1 else 0.0
        max_amount[u] = all_amounts.max() if len(all_amounts) > 0 else 0.0
        min_amount[u] = all_amounts.min() if len(all_amounts) > 0 else 0.0
        out_deg[u] = len(outs)
        in_deg[u] = len(ins)

    total_deg = in_deg + out_deg
    log_sum_amt = np.log1p(sum_amount)
    ratio_in_out = np.divide(in_deg + 1, out_deg + 1)
    feats = np.vstack([
        in_deg, out_deg, total_deg, sum_amount, mean_amount, std_amount,
        max_amount, min_amount, log_sum_amt, ratio_in_out
    ]).T
    scaler = StandardScaler()
    feats = scaler.fit_transform(feats)
    edge_index_torch = torch.tensor(edge_index, dtype=torch.long)
    undirected_edge_index = to_undirected(edge_index_torch)
    x = torch.tensor(feats, dtype=torch.float)
    data = Data(x=x, edge_index=undirected_edge_index)
    data.num_nodes = num_nodes
    mapping = {"node_to_idx": node_to_idx, "idx_to_node": idx_to_node, "scaler": scaler}
    return data, mapping

class GCNEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 128, out_channels: int = 64, dropout: float = 0.4):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout
        self.leaky = nn.LeakyReLU(0.1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.leaky(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.leaky(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        z = self.conv_mu(x, edge_index)
        return z

# -------------------------
# TRAIN/LOAD function (with metrics)
# -------------------------
@st.cache_resource(show_spinner=False)
def load_or_train_model(
    df: pd.DataFrame,
    force_retrain: bool = False,
    epochs: int = 100,
    lr: float = 0.001,
    hidden: int = 256,
    emb: int = 128,
    device_str: str = "cpu",
    custom_name: str = "",
    patience: int = 20,
):
    ensure_dirs()
    fingerprint = dataset_fingerprint(df)
    model_path = Path("models_path") / f"{fingerprint}_gagnn.pth"
    pred_path = Path("models_path") / f"{fingerprint}_node_predictions.csv"
    meta_path = Path("models_path") / f"{fingerprint}_meta.json"

    custom_model_path = Path("models_path") / f"{custom_name}.pth" if custom_name else None
    custom_pred_path = Path("models_path") / f"{custom_name}_node_predictions.csv" if custom_name else None

    fixed_model_path = Path("models_path") / "gagnn_model.pth"
    fixed_pred_path = Path("models_path") / "node_predictions.csv"

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # =========================
    # 1) LOAD SAVED MODEL PATH
    # =========================
    if model_path.exists() and pred_path.exists() and not force_retrain:
        st.info("Found saved model & predictions — loading them.")
        preds_df = pd.read_csv(pred_path)

        # default None metrics
        metrics = {
            "Accuracy": None,
            "Precision": None,
            "Recall": None,
            "F1 Score": None,
            "AUC-ROC": None,
        }

        # try to load metrics from meta file (if present)
        if meta_path.exists():
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                # if meta stored as {"metrics": {...}}
                if isinstance(meta, dict) and "metrics" in meta:
                    loaded_metrics = meta["metrics"]
                else:
                    # or meta IS the metrics dict itself
                    loaded_metrics = meta
                for k in metrics.keys():
                    if k in loaded_metrics and loaded_metrics[k] is not None:
                        metrics[k] = loaded_metrics[k]
            except Exception:
                pass  # keep default None values if anything fails

        return preds_df, str(model_path), metrics

    # =========================
    # 2) TRAIN NEW MODEL
    # =========================
    st.info("No saved model found for this dataset — starting unsupervised training (GAE + clustering).")
    data, mapping = build_graph_from_df(df)
    data = train_test_split_edges(data)
    data = data.to(device)

    in_ch = data.x.size(1)
    encoder = GCNEncoder(in_ch, hidden_channels=hidden // 2, out_channels=emb, dropout=0.4)
    model = GAE(encoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    progress = st.progress(0)
    status = st.empty()
    start_time = time.time()
    best_val = float("inf")
    patience_count = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        z = model.encode(data.x, data.train_pos_edge_index)

        pos_edge_index = data.train_pos_edge_index
        pos_out = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1)

        neg_edge_index = negative_sampling(
            edge_index=data.train_pos_edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=pos_edge_index.size(1),
        ).to(device)
        neg_out = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)

        logits = torch.cat([pos_out, neg_out], dim=0)
        labels = torch.cat(
            [
                torch.ones(pos_out.size(0), device=device),
                torch.zeros(neg_out.size(0), device=device),
            ],
            dim=0,
        )

        loss = F.binary_cross_entropy_with_logits(logits, labels)
        reg = 1e-4 * (z.norm(p=2).pow(2)) / z.numel()
        total_loss = loss + reg
        total_loss.backward()
        optimizer.step()

        val_estimate = total_loss.item()
        progress.progress(epoch / epochs)
        status.text(f"Epoch {epoch}/{epochs} — loss: {total_loss.item():.6f}")

        if val_estimate < best_val - 1e-6:
            best_val = val_estimate
            patience_count = 0
            torch.save({"model_state_dict": model.state_dict(), "mapping": mapping}, str(model_path))
        else:
            patience_count += 1
            if patience_count >= patience:
                status.text(f"Early stopping after {epoch} epochs (no improvement).")
                break

    # --- Evaluation on train edges (as proxy since unsupervised) ---
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.train_pos_edge_index)

        pos_edge_index = data.train_pos_edge_index
        pos_out = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1)
        pos_probs = torch.sigmoid(pos_out).cpu().numpy()
        pos_labels = np.ones(pos_probs.shape[0])

        neg_edge_index = negative_sampling(
            edge_index=data.train_pos_edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=pos_edge_index.size(1),
        ).to(device)
        neg_out = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)
        neg_probs = torch.sigmoid(neg_out).cpu().numpy()
        neg_labels = np.zeros(neg_probs.shape[0])

        y_true = np.concatenate([pos_labels, neg_labels])
        y_pred_prob = np.concatenate([pos_probs, neg_probs])
        y_pred = (y_pred_prob >= 0.5).astype(int)

        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
        )

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred_prob)

        metrics = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "AUC-ROC": auc,
        }

    st.subheader("Evaluation Metrics")
    st.table(pd.DataFrame(metrics, index=[0]).T.rename(columns={0: "Score"}))

    elapsed = time.time() - start_time

    # Final embeddings & clustering
    model.eval()
    with torch.no_grad():
        final_z = model.encode(data.x, data.train_pos_edge_index)
        Z = final_z.cpu().numpy()

    n_clusters = min(30, max(2, int(np.sqrt(data.num_nodes))))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    groups = kmeans.fit_predict(Z)
    centroids = kmeans.cluster_centers_
    dists = np.linalg.norm(Z - centroids[groups], axis=1)
    anomaly_scores = MinMaxScaler().fit_transform(dists.reshape(-1, 1)).flatten()
    fraud_prob = anomaly_scores

    idx_to_node = mapping["idx_to_node"]
    preds_df = pd.DataFrame(
        {
            "node_idx": np.arange(data.num_nodes),
            "node": [idx_to_node[i] for i in range(data.num_nodes)],
            "group": groups,
            "anomaly_score": anomaly_scores,
            "fraud_prob": fraud_prob,
        }
    )

    # Save artifacts
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "kmeans_centers": centroids,
            "mapping": mapping,
            "params": {"hidden": hidden, "emb": emb},
        },
        str(model_path),
    )
    preds_df.to_csv(pred_path, index=False)

    meta = {
        "n_nodes": int(data.num_nodes),
        "n_clusters": int(n_clusters),
        "metrics": metrics,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "kmeans_centers": centroids,
            "mapping": mapping,
            "params": {"hidden": hidden, "emb": emb},
        },
        str(fixed_model_path),
    )
    preds_df.to_csv(str(fixed_pred_path), index=False)

    if custom_name:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "kmeans_centers": centroids,
                "mapping": mapping,
                "params": {"hidden": hidden, "emb": emb},
            },
            str(custom_model_path),
        )
        preds_df.to_csv(str(custom_pred_path), index=False)

    st.success(f"Training complete in {elapsed:.2f} seconds.")
    st.write("Saved artifacts:")
    st.write(f"- Fingerprint model: `{model_path}`")
    st.write(f"- Fingerprint predictions: `{pred_path}`")
    st.write(f"- Fixed model: `{fixed_model_path}`")
    st.write(f"- Fixed predictions: `{fixed_pred_path}`")
    if custom_name:
        st.write(f"- Custom model: `{custom_model_path}`")
        st.write(f"- Custom predictions: `{custom_pred_path}`")

    return preds_df, str(model_path), metrics



# -------------------------
# Plotting helper
# -------------------------
def plot_graph_with_predictions(df: pd.DataFrame, preds_df: pd.DataFrame, src_col="source", tgt_col="target", amount_col="amount", mode="Interactive", threshold=0.5):
    node_to_prob = {r["node"]: r["fraud_prob"] for _, r in preds_df.iterrows()}
    node_to_group = {r["node"]: int(r["group"]) for _, r in preds_df.iterrows()}

    if src_col in df.columns and tgt_col in df.columns:
        G = nx.from_pandas_edgelist(df, source=src_col, target=tgt_col, edge_attr=amount_col, create_using=nx.DiGraph())
    else:
        raise ValueError("Dataframe must have source/target columns")

    pos = nx.spring_layout(G, seed=42)
    edge_x, edge_y = [], []
    for e in G.edges():
        x0, y0 = pos[e[0]]
        x1, y1 = pos[e[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')

    node_x, node_y, node_text, node_color = [], [], [], []
    node_size = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        prob = float(node_to_prob.get(node, 0.0))
        grp = int(node_to_group.get(node, -1))
        connected = list(G.edges(node, data=True))
        total_amount = sum([edata.get(amount_col, 0) for _, _, edata in connected])
        txt = f"User: {node}<br>Group: {grp}<br>FraudProb: {prob:.3f}<br>TotalAmount: {total_amount:.2f}"
        node_text.append(txt)
        node_color.append(prob)
        node_size.append(8 + 24 * prob)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[str(n) for n in G.nodes()],
        hoverinfo='text',
        hovertext=node_text,
        textposition="top center",
        marker=dict(
            showscale=True,
            colorscale='RdYlBu_r',
            color=node_color,
            size=node_size,
            colorbar=dict(title="Fraud Prob", x=1.02),
            line=dict(width=1, color='DarkSlateGrey')
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="GAGNN: Predicted Fraud Groups & Probabilities",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )

    high_nodes = [n for n, p in node_to_prob.items() if p >= threshold]
    return fig, high_nodes

# -------------------------
# Streamlit app UI
# -------------------------
def main():
    st.set_page_config(layout="wide")
    st.title("Anti Money laundering System using GAGNN — (Unsupervised)")

    st.sidebar.header("Controls")
    uploaded_file = st.sidebar.file_uploader("Upload CSV (transactions)", type=["csv"])
    use_default = st.sidebar.checkbox("Use default converted01.csv if present (data/converted01.csv)")
    custom_name = st.sidebar.text_input("Custom save-name (no extension)", value="")
    epochs = st.sidebar.number_input("Training epochs (if needed)", min_value=10, max_value=2000, value=200, step=10)
    # lr = st.sidebar.number_input("Learning rate", min_value=1e-5, max_value=1e-1, value=1e-3, format="%.6f")
    # hidden = st.sidebar.number_input("Hidden dimension (GCN)", min_value=64, max_value=1024, value=256, step=64)
    # emb = st.sidebar.number_input("Embedding dim", min_value=16, max_value=512, value=128, step=16)
    force_retrain = st.sidebar.checkbox("Force retrain (ignore saved model)")
    threshold = st.sidebar.slider("Suspicion threshold (probability)", 0.0, 1.0, 0.5, 0.01)
    device_choice = st.sidebar.selectbox("Device", ["cpu", "cuda"])
    patience = st.sidebar.number_input("Early stop patience", min_value=1, max_value=100, value=10)

    st.markdown("""
    **Notes**
    - First run trains an unsupervised GAE (may take time).
    - Trained model and predictions are saved under `models_path/` with a dataset fingerprint.
    - Subsequent runs load saved predictions for fast visualization.
    """)

    df = None
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif use_default:
        if os.path.exists("data/converted01.csv"):
            df = pd.read_csv("data/converted01.csv")
            st.info("Loaded data/converted01.csv")
        elif os.path.exists("data/transactions01.csv"):
            df = pd.read_csv("data/transactions01.csv")
            st.info("Loaded data/transactions01.csv")

    if df is None:
        st.warning("Please upload a transactions CSV or enable 'Use default'. App awaits data.")
        st.stop()

    st.subheader("Sample of uploaded data")
    st.dataframe(df.head())

    if "nameOrig" in df.columns and "nameDest" in df.columns:
        df = df.rename(columns={"nameOrig": "source", "nameDest": "target"})
    if "source" not in df.columns or "target" not in df.columns:
        st.error("CSV must contain 'source' and 'target' columns (or nameOrig/nameDest).")
        st.stop()

    preds_df, model_path, metrics = load_or_train_model(
        df,
        force_retrain=force_retrain,
        epochs=int(epochs),
        lr=0.001, #float(lr),
        hidden=256, #int(hidden),
        emb=128, #int(emb),
        device_str=device_choice,
        custom_name=custom_name,
        patience=int(patience)
    )


    st.success(f"Model ready — loaded/saved at: {model_path}")

    #st.subheader("Model Accuracy")
    #st.metric("Accuracy", f"{metrics['Accuracy']:.2%}")

    # st.subheader("Link Reconstruction Accuracy")

    # acc_val = metrics.get("Accuracy")
    # if acc_val is not None:
    #     st.metric("Accuracy", f"{acc_val:.2%}")
    # else:
    #     st.metric("Accuracy", "N/A (loaded model)")



    st.write(f"Predictions: {preds_df.shape[0]} nodes")

    st.subheader("Interactive Graph (predictions)")
    fig, high_nodes = plot_graph_with_predictions(df, preds_df, threshold=threshold)
    st.plotly_chart(fig, use_container_width=True, height=700)

   # st.subheader("Extra Tree Graph")
    G = nx.from_pandas_edgelist(df, source='source', target='target', edge_attr='amount', create_using=nx.DiGraph())
    # fig_tree = visualize_tree_graph_plotly(G, set(preds_df[preds_df['fraud_prob']>threshold]['node']))
    # st.plotly_chart(fig_tree, use_container_width=True)

    st.subheader("Static Tree Graph")
    fraud_nodes = set(
    n for n in preds_df[preds_df["fraud_prob"] >= threshold]["node"]
    if n in G.nodes()
)
    G = nx.from_pandas_edgelist(df, source="source", target="target",
                            edge_attr="amount", create_using=nx.DiGraph())
    plt_fig = visualize_tree_graph_static(G, fraud_nodes)
    st.pyplot(plt_fig)


    st.subheader("Summary Table — All Nodes (click headers to sort)")
    preds_sorted = preds_df.sort_values("fraud_prob", ascending=False).reset_index(drop=True)
    st.dataframe(preds_sorted)

    st.subheader(f"High-risk nodes (fraud_prob >= {threshold:.2f})")
    high_df = preds_df[preds_df["fraud_prob"] >= threshold].sort_values("fraud_prob", ascending=False).reset_index(drop=True)
    st.dataframe(high_df)

    fp_fingerprint = Path("models_path") / f"{dataset_fingerprint(df)}_node_predictions.csv"
    if fp_fingerprint.exists():
        with open(fp_fingerprint, "rb") as f:
            st.download_button("Download node predictions (fingerprint)", data=f, file_name=fp_fingerprint.name, mime="text/csv")
    fp_fixed = Path("models_path") / "node_predictions.csv"
    if fp_fixed.exists():
        with open(fp_fixed, "rb") as f:
            st.download_button("Download node_predictions.csv (fixed name)", data=f, file_name=fp_fixed.name, mime="text/csv")
    mp_fixed = Path("models_path") / "gagnn_model.pth"
    if mp_fixed.exists():
        with open(mp_fixed, "rb") as f:
            st.download_button("Download model (gagnn_model.pth)", data=f, file_name=mp_fixed.name, mime="application/octet-stream")
    if custom_name:
        mp_custom = Path("models_path") / f"{custom_name}.pth"
        pp_custom = Path("models_path") / f"{custom_name}_node_predictions.csv"
        if pp_custom.exists():
            with open(pp_custom, "rb") as f:
                st.download_button(f"Download predictions ({custom_name})", data=f, file_name=pp_custom.name, mime="text/csv")
        if mp_custom.exists():
            with open(mp_custom, "rb") as f:
                st.download_button(f"Download model ({custom_name}.pth)", data=f, file_name=mp_custom.name, mime="application/octet-stream")

    st.info("Done — next runs will load the saved model/predictions for this dataset.")

    st.markdown(
        """
        <style>
        .footer {
            position: fixed;
            bottom: 0;
            right: 0;
            padding: 5px 10px;
            font-size: 12px;
            color: rgba(100, 100, 100, 0.7);
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
