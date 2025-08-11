import torch
def train(model, classifier, data, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        node_embeddings = model(data.x, data.edge_index)
        node_preds = classifier(node_embeddings[data.edge_index[0]])
        loss = criterion(node_preds, data.y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

