def joint_loss(node_pred, node_label, edge_pred, edge_label, group_pred=None, group_label=None,
               lambda_node=1.0, lambda_edge=1.0, lambda_group=1.0):
    import torch.nn.functional as F

    loss_node = F.cross_entropy(node_pred, node_label)
    loss_edge = F.cross_entropy(edge_pred, edge_label)

    if group_pred is not None and group_label is not None:
        loss_group = F.cross_entropy(group_pred, group_label)
    else:
        loss_group = 0.0

    total_loss = lambda_node * loss_node + lambda_edge * loss_edge + lambda_group * loss_group
    return total_loss
