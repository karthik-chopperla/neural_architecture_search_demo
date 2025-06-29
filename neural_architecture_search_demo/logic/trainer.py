# Placeholder for logic/trainer.py
# logic/trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

def train_model(model, X_train, y_train, X_val, y_val, epochs=10, lr=0.01):
    """
    Train the given model and evaluate on validation data.

    Returns:
        dict: Contains model, training/validation accuracy and loss
    """
    device = torch.device("cpu")
    model.to(device)

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.long).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val).argmax(dim=1)
        val_acc = accuracy_score(y_val.cpu(), val_preds.cpu())
    
    return {
        "model": model,
        "val_accuracy": val_acc
    }
