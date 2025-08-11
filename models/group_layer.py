from networkx import Graph
import networkx as nx

# Forms groups of users based on high-risk transactions
def group_aggregation(edge_index, edge_scores, threshold=0.9):
    selected_edges = edge_scores >= threshold
    edge_subset = edge_index[:, selected_edges]

    G = nx.Graph()
    G.add_edges_from(edge_subset.t().tolist())
    components = list(nx.connected_components(G))

    node_to_group = {}
    for group_id, group in enumerate(components):
        for node in group:
            node_to_group[node] = group_id

    return node_to_group  # node_id -> group_id
