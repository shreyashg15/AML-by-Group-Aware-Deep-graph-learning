import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Load your converted CSV
df = pd.read_csv("data/converted01.csv")  # change path if needed

# Map users to node IDs
user_ids = pd.concat([df["source"], df["target"]]).astype(str).unique()
user_to_id = {user: i for i, user in enumerate(user_ids)}
id_to_user = {i: user for user, i in user_to_id.items()}

# Build graph
G = nx.DiGraph()
for _, row in df.iterrows():
    src = user_to_id[str(row["source"])]
    dst = user_to_id[str(row["target"])]
    label = int(row["label"])

    G.add_node(src, label=str(row["source"]), is_fraud=0)
    G.add_node(dst, label=str(row["target"]), is_fraud=label)
    G.add_edge(src, dst, is_fraud=label)

# Plot graph
pos = nx.spring_layout(G, seed=42)
fig, ax = plt.subplots(figsize=(10, 8))
for node in G.nodes():
    is_fraud = G.nodes[node]["is_fraud"]
    color = "red" if is_fraud else "green"
    label = G.nodes[node]["label"]
    nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color=color, node_size=600, ax=ax)
    nx.draw_networkx_labels(G, pos, labels={node: label}, font_color="white", font_size=8, ax=ax)

nx.draw_networkx_edges(G, pos, arrowstyle="-|>", arrows=True, edge_color="gray", ax=ax)

legend = [
    Patch(facecolor="green", edgecolor="black", label="🟩 Non-Fraud"),
    Patch(facecolor="red", edgecolor="black", label="🟥 Fraud"),
]
ax.legend(handles=legend, title="User Type")
ax.set_title("Transaction Graph")
ax.axis("off")
fig.tight_layout()

# Streamlit render
st.title("✅ Simple Transaction Graph")
st.pyplot(fig)
