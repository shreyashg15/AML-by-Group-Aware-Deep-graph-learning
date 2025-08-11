# utils/visualize.py
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st
from pyvis.network import Network
import tempfile
import os
import torch
from matplotlib.patches import Patch

def visualize_graph(data,
                    groups,
                    fraud_labels=None,
                    user_ids=None,
                    mode="static",
                    max_groups_to_plot=10,
                    transaction_info=None,
                    height=750):
    """
    data: PyG Data or edge_index-like (expects data.edge_index)
    groups: dict node->group OR list/iterable
    fraud_labels: edge-level labels (torch tensor or list) — used to mark nodes touching fraud edges
    user_ids: dict node->original id (optional)
    mode: "static" (matplotlib) or "interactive" (pyvis)
    transaction_info: dict {(src, dst): amount} for edge hover in interactive mode
    """

    # normalize edge list
    if hasattr(data, "edge_index"):
        edge_index = data.edge_index
    else:
        edge_index = data

    if isinstance(edge_index, torch.Tensor):
        edges = edge_index.cpu().long().t().tolist()
    else:
        # assume numpy or list-like with shape [2, E]
        try:
            edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
        except Exception:
            # fallback: iterable of pairs
            edges = [tuple(e) for e in edge_index]

    # Build networkx graph (directed for interactive arrows; for static we draw without arrows for clarity)
    G_directed = nx.DiGraph()
    G_directed.add_edges_from(edges)

    # Build node->group mapping
    if groups is None:
        groups_map = {}
    elif isinstance(groups, dict):
        groups_map = {int(k): int(v) for k, v in groups.items()}
    elif isinstance(groups, torch.Tensor):
        groups_map = {i: int(g) for i, g in enumerate(groups.cpu().tolist())}
    else:
        # list-like mapping by index
        try:
            groups_map = {i: int(g) for i, g in enumerate(list(groups))}
        except Exception:
            groups_map = {}

    # Determine fraud nodes (nodes involved in any fraud edge)
    fraud_nodes = set()
    if fraud_labels is not None:
        # ensure list
        if isinstance(fraud_labels, torch.Tensor):
            fl = fraud_labels.cpu().tolist()
        else:
            fl = list(fraud_labels)
        for (s, d), lbl in zip(edges, fl):
            try:
                if int(lbl) != 0:
                    fraud_nodes.add(int(s)); fraud_nodes.add(int(d))
            except Exception:
                continue

    # Map user_ids
    if user_ids is None:
        # try from data
        user_ids = getattr(data, "user_map", None)
    # ensure user_ids is dict-like str
    if user_ids is None:
        user_ids = {n: str(n) for n in range(getattr(data, "num_nodes", max(max(edges)[0], max(edges)[1]) + 1 if edges else 0))}

    # Limit groups to plot
    unique_groups = sorted(set(groups_map.values())) if groups_map else []
    if max_groups_to_plot and len(unique_groups) > max_groups_to_plot:
        unique_groups = unique_groups[:max_groups_to_plot]
        keep_nodes = {n for n, g in groups_map.items() if g in set(unique_groups)}
        edges = [e for e in edges if e[0] in keep_nodes and e[1] in keep_nodes]
        G_directed = nx.DiGraph()
        G_directed.add_edges_from(edges)

    # STATIC: matplotlib
    if mode == "static":
        G = nx.Graph()
        G.add_edges_from(edges)
        pos = nx.spring_layout(G, seed=42)

        fig, ax = plt.subplots(figsize=(12, 9))

        # node facecolors (group-based) and edgecolors for fraud
        group_colors = {}
        cmap = plt.get_cmap("tab20")
        for i, gid in enumerate(unique_groups):
            group_colors[gid] = cmap(i % 20)

        node_face = []
        node_edge = []
        labels = {}
        for n in G.nodes():
            g = groups_map.get(n, None)
            if g is not None and g in group_colors:
                node_face.append(group_colors[g])
            else:
                node_face.append("lightgrey")
            node_edge.append("red" if n in fraud_nodes else "black")
            labels[n] = str(user_ids.get(n, n))

        nx.draw_networkx_nodes(G, pos, node_color=node_face, edgecolors=node_edge, linewidths=1.5, node_size=500, ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.6, ax=ax)
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)

        # legend
        legend_patches = [Patch(color=col, label=f"Group {gid}") for gid, col in group_colors.items()]
        legend_patches.append(Patch(edgecolor='red', facecolor='white', label="Fraud Node", linewidth=2))
        legend_patches.append(Patch(edgecolor='black', facecolor='white', label="Non-Fraud Node", linewidth=2))
        ax.legend(handles=legend_patches, title="Group Legend", loc="upper right", fontsize="small")

        ax.set_title("Detected Groups")
        ax.axis("off")
        st.pyplot(fig)
        return fig

    # INTERACTIVE: PyVis
    elif mode == "interactive":
        net = Network(height=f"{height}px", width="100%", directed=True, bgcolor="#ffffff", font_color="black")
        net.toggle_physics(True)

        # add nodes
        nodes_in_graph = set()
        for s, d in edges:
            nodes_in_graph.add(int(s)); nodes_in_graph.add(int(d))
        for n in nodes_in_graph:
            gid = groups_map.get(n, None)
            title = f"User: {user_ids.get(n, n)}<br>Node: {n}"
            if n in fraud_nodes:
                title += "<br><b>Potential fraud node</b>"

            color = None
            if gid is not None:
                # use a color by group index (simple hash)
                color = plt.get_cmap("tab20")(int(gid) % 20)
                # convert rgba to hex
                from matplotlib.colors import to_hex
                color = to_hex(color)
            else:
                color = "#DDDDDD"

            net.add_node(n, label=str(user_ids.get(n, n)), title=title, color=color, borderWidth=2)

        # add edges with arrows and titles (amount)
        for s, d in edges:
            amt = None
            if transaction_info:
                amt = transaction_info.get((int(s), int(d)), transaction_info.get((str(s), str(d)), None))
            ttl = f"From: {user_ids.get(s, s)} → To: {user_ids.get(d, d)}"
            if amt is not None:
                ttl += f"<br>Amount: {amt}"
            net.add_edge(int(s), int(d), title=ttl)

        # Save to temporary file and embed in Streamlit
        tmp_dir = tempfile.gettempdir()
        out_file = os.path.join(tmp_dir, "pyvis_graph.html")
        net.save_graph(out_file)
        with open(out_file, "r", encoding="utf-8") as f:
            html = f.read()
        # embed
        st.components.v1.html(html, height=height, scrolling=True)
        # keep file (no remove) - optional cleanup can be done later
        return None

    else:
        raise ValueError("Unknown mode for visualize_graph: choose 'static' or 'interactive'")
