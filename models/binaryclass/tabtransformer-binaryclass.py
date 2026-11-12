import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import pandas as pd
import numpy as np 

# TabTransformer model
class TabTransformer(nn.Module):
    def __init__(self, num_features, dim_embedding=128, num_heads=4, num_layers=5):
        super(TabTransformer, self).__init__()
        self.embedding = nn.Linear(num_features, dim_embedding)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_embedding, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(dim_embedding, 1) 

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)  
        x = self.transformer(x)
        x = torch.mean(x, dim=1)  
        x = self.classifier(x)
        return x 

def evaluate(model, loader, device):
    """Evaluate exactly one split with fresh prediction accumulators."""
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    targets, probabilities = [], []
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device).reshape(-1, 1)
            logits = model(features).reshape(-1, 1)
            total_loss += criterion(logits, labels).item() * labels.shape[0]
            targets.extend(labels.cpu().numpy().reshape(-1))
            probabilities.extend(torch.sigmoid(logits).cpu().numpy().reshape(-1))
    if not targets:
        raise ValueError("Evaluation requires a non-empty split.")
    predictions = np.asarray(probabilities) > 0.5
    return {
        'loss': total_loss / len(targets),
        'accuracy': accuracy_score(targets, predictions),
        'precision': precision_score(targets, predictions, zero_division=0),
        'recall': recall_score(targets, predictions, zero_division=0),
        'auc': roc_auc_score(targets, probabilities),
        'n_samples': len(targets),
    }


def train_model(model, train_loader, val_loader, test_loader, num_epochs=300,
                learning_rate=0.0001, patience=5, device=None,
                checkpoint_path="best_model.pth"):
    """Train, select a checkpoint on validation loss, then evaluate the test split."""
    if num_epochs < 1 or len(train_loader) == 0:
        raise ValueError("Training requires a positive epoch count and non-empty data.")
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Preserve the original binary transformer learning-rate schedule.
    def lr_lambda(epoch):
        if epoch < 50:
            return 1.0
        elif epoch < 75:
            return 0.01
        return 0.001

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'learning_rates': []}
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_samples = 0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device).reshape(-1, 1)
            optimizer.zero_grad()
            logits = model(features).reshape(-1, 1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * labels.shape[0]
            train_samples += labels.shape[0]
        train_loss /= train_samples
        validation = evaluate(model, val_loader, device)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(validation['loss'])
        history['learning_rates'].append(scheduler.get_last_lr()[0])
        print(f"Epoch {epoch}, Train Loss: {train_loss:.6f}, "
              f"Val Loss: {validation['loss']:.6f}, "
              f"Val Accuracy: {validation['accuracy']:.6f}, "
              f"Precision: {validation['precision']:.6f}, "
              f"Recall: {validation['recall']:.6f}, AUC: {validation['auc']:.6f}")
        if validation['loss'] < best_val_loss:
            best_val_loss = validation['loss']
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break
        scheduler.step()

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    # Each evaluate() call owns its lists, so validation samples cannot enter test metrics.
    metrics = evaluate(model, test_loader, device)
    return model, history, metrics


def run_experiment(betti_train, labels_train, betti_val, labels_val,
                   betti_test, labels_test, batch_size=128, **training_options):
    """Run the original 400-feature model using preloaded arrays or DataFrames."""
    def loader(features, labels, shuffle):
        dataset = TensorDataset(
            torch.as_tensor(np.asarray(features), dtype=torch.float32),
            torch.as_tensor(np.asarray(labels), dtype=torch.float32).reshape(-1, 1),
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    model, history, metrics = train_model(
        TabTransformer(num_features=400),
        loader(betti_train, labels_train, True),
        loader(betti_val, labels_val, False),
        loader(betti_test, labels_test, False),
        **training_options,
    )
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test AUC: {metrics['auc']:.4f}")
    print(f"Test Precision: {metrics['precision']:.4f}")
    print(f"Test Recall: {metrics['recall']:.4f}")
    return model, history, metrics


if __name__ == "__main__":
    model, history, metrics = run_experiment(
        betti_train, labels_train, betti_val, labels_val, betti_test, labels_test,
    )
