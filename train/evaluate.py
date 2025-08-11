from sklearn.metrics import roc_auc_score, precision_score, recall_score

def evaluate(preds, labels):
    preds_soft = preds.softmax(dim=1)[:, 1].detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    auc = roc_auc_score(labels, preds_soft)

    return {
        'AUC': auc
    }

def precision_at_k(preds, labels, k_percent=0.01):
    preds_soft = preds.softmax(dim=1)[:, 1].detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    k = int(len(preds_soft) * k_percent)
    topk_idx = preds_soft.argsort()[::-1][:k]

    precision = precision_score(labels[topk_idx], [1]*k)
    recall = recall_score(labels[topk_idx], [1]*k)

    return {
        'P@K': precision,
        'R@K': recall
    }
