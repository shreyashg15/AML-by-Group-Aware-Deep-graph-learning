import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# =============================
# Graph Visualization Function
# =============================
def visualize_graph(G, fraud_nodes, pos, groups, mode="Static", transactions=None):
    if mode == "Static":
        plt.figure(figsize=(12, 8))
        nx.draw_networkx_edges(G, pos, alpha=0.5)

        # Draw fraud nodes with red border
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=[n for n in G.nodes if n in fraud_nodes],
            node_color='lightcoral',
            edgecolors='red',
            node_size=500
        )

        # Draw non-fraud nodes with black border
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=[n for n in G.nodes if n not in fraud_nodes],
            node_color='lightblue',
            edgecolors='black',
            node_size=500
        )

        nx.draw_networkx_labels(G, pos, font_size=8)
        plt.title("Static Fraud Graph", fontsize=15)
        st.pyplot(plt)

    elif mode == "Interactive":
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

        # Prepare hover info
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            # Calculate total transaction amount for this node
            connected_edges = list(G.edges(node, data=True))
            total_amount = sum([edata.get('amount', 0) for _, _, edata in connected_edges])

            if node in fraud_nodes:
                node_color.append('lightcoral')
                node_border_color.append('red')
            else:
                node_color.append('lightblue')
                node_border_color.append('black')

            hover_text.append(f"User ID: {node}<br>Total Amount: {total_amount}")

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

        st.plotly_chart(fig, use_container_width=True)


# =============================
# Extra Tree Graph Function
# =============================
def visualize_tree_graph(G, fraud_nodes):
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)

    nx.draw_networkx_edges(G, pos, alpha=0.5, arrows=True, arrowstyle='-|>', arrowsize=12)

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=[n for n in G.nodes if n in fraud_nodes],
        node_color='lightcoral',
        edgecolors='red',
        node_size=500
    )
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=[n for n in G.nodes if n not in fraud_nodes],
        node_color='lightblue',
        edgecolors='black',
        node_size=500
    )

    nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title("Tree Graph - Money Laundering Flow", fontsize=15)
    fraud_patch = plt.Line2D([0], [0], marker='o', color='w', label='Fraud',
                             markerfacecolor='lightcoral', markeredgecolor='red', markersize=10)
    nonfraud_patch = plt.Line2D([0], [0], marker='o', color='w', label='Non-Fraud',
                                markerfacecolor='lightblue', markeredgecolor='black', markersize=10)
    plt.legend(handles=[fraud_patch, nonfraud_patch])
    st.pyplot(plt)


# =============================
# Main App
# =============================
def main():
    st.set_page_config(layout="wide")
    st.title("Anti-Money Laundering Detector")

    # Sidebar controls
    st.sidebar.header("Controls")

    # Upload CSV moved to top
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    use_default = st.sidebar.checkbox("Use default converted_transactions.csv if present")
    converted_file_name = st.sidebar.text_input("Converted file name (no extension)", value="converted0")
    threshold = st.sidebar.slider("Suspicion Threshold", 0.0, 1.0, 0.5, 0.01)
    max_groups = st.sidebar.slider("Max Groups to Visualize", 1, 50, 15)
    mode = st.sidebar.selectbox("Graph Mode", ["Interactive", "Static"])
    epochs = st.sidebar.number_input("Training epochs (small=fast)", min_value=1, max_value=500, value=30, step=1)

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Sample Data")
        st.dataframe(df.head())

        # Build Graph
        if 'nameOrig' in df.columns and 'nameDest' in df.columns:
            df = df.rename(columns={'nameOrig': 'source', 'nameDest': 'target'})
        
        if 'label' not in df.columns and 'isFraud' in df.columns:
            df['label'] = df['isFraud']

        G = nx.from_pandas_edgelist(df, source='source', target='target', edge_attr='amount', create_using=nx.DiGraph())
        fraud_nodes = set(df[df['label'] == 1]['source']).union(set(df[df['label'] == 1]['target']))
        pos = nx.spring_layout(G)

        st.subheader("Main Graph")
        visualize_graph(G, fraud_nodes, pos, groups=None, mode=mode, transactions=df)

        st.subheader("Extra Tree Graph")
        visualize_tree_graph(G, fraud_nodes)


if __name__ == "__main__":
    main()
