import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, roc_auc_score
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

# Initialize model, loss, optimizer, and device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TabTransformer(num_features=400).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Convert training and validation data to tensors and DataLoader
X_train_tensor = torch.FloatTensor(betti_train.values).to(device)
y_train_tensor = torch.FloatTensor(labels_train).unsqueeze(1).to(device)  

X_val_tensor = torch.FloatTensor(betti_val.values).to(device)
y_val_tensor = torch.FloatTensor(labels_val).unsqueeze(1).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# Early stopping 
patience = 5  
best_val_loss = float('inf') 
patience_counter = 0

# learning rate scheduler
def lr_lambda(epoch):
    if epoch < 50:
        return 1.0  # 0.0001 * 1.0 = 0.0001
    elif epoch < 75:
        return 0.01  # 0.001 * 0.01 = 0.00001
    else:
        return 0.001  # 0.001 * 0.001 = 0.000001

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Training loop 
num_epochs = 300
for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch) 
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)

    # Validation 
    model.eval()
    val_loss = 0
    all_targets = []
    all_predictions = []
    all_probs = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            output = model(X_batch)
            loss = criterion(output, y_batch)
            val_loss += loss.item()
            probs = torch.sigmoid(output)  
            predictions = (probs > 0.5).int()
            all_targets.extend(y_batch.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    val_loss /= len(val_loader)
    val_accuracy = (torch.tensor(all_predictions) == torch.tensor(all_targets)).sum().item() / len(all_targets)

    # Calculate metrics
    precision = precision_score(all_targets, all_predictions)
    recall = recall_score(all_targets, all_predictions)
    auc = roc_auc_score(all_targets, all_probs)

    # Print
    print(f"Epoch {epoch}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
          f"Val Accuracy: {val_accuracy:.6f}, Precision: {precision:.6f}, Recall: {recall:.6f}, AUC: {auc:.6f}")

    # Early stopping logic
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0  
        # Save best model
        torch.save(model.state_dict(), "best_model.pth")
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping triggered.")
        break

    # Step scheduler
    scheduler.step()

# Load best model after training
model.load_state_dict(torch.load("best_model.pth"))

# Evaluation
model.eval()
X_test_tensor = torch.FloatTensor(betti_test.values).to(device)
y_test_tensor = torch.FloatTensor(labels_test).unsqueeze(1).to(device) 

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        output = model(X_batch) 
        probs = torch.sigmoid(output) 
        predictions = (probs > 0.5).float() 

        all_targets.extend(y_batch.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# Calculate metrics
accuracy = (torch.tensor(all_predictions) == torch.tensor(all_targets)).float().mean().item()
precision = precision_score(all_targets, all_predictions)
recall = recall_score(all_targets, all_predictions)
auc = roc_auc_score(all_targets, all_probs)

# Print metrics
print(f'Test Accuracy: {accuracy:.4f}')
print(f'Test AUC: {auc:.4f}')
print(f'Test Precision: {precision:.4f}')
print(f'Test Recall: {recall:.4f}')
