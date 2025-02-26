import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import numpy as np
from tensorflow.keras.utils import to_categorical
import pandas as pd 

num_classes = 5  # 3 for NIAID

labels_train = to_categorical(labels_train, num_classes=num_classes)
labels_val = to_categorical(labels_val, num_classes=num_classes)
labels_test = to_categorical(labels_test, num_classes=num_classes)

# TabTransformer for multiclass
class TabTransformer(nn.Module):
    def __init__(self, num_features, num_classes=num_classes, dim_embedding=128, num_heads=4, num_layers=5):  
        super(TabTransformer, self).__init__()
        self.embedding = nn.Linear(num_features, dim_embedding)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_embedding, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(dim_embedding, num_classes)  

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = torch.mean(x, dim=1)
        x = self.classifier(x)
        return x

# Initialize model, loss, optimizer, and device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TabTransformer(num_features=400, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Convert data to tensors
X_train_tensor = torch.FloatTensor(betti_train.values).to(device)
y_train_tensor = torch.FloatTensor(labels_train).to(device)

X_val_tensor = torch.FloatTensor(betti_val.values).to(device)
y_val_tensor = torch.FloatTensor(labels_val).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# Learning rate scheduler
def lr_lambda(epoch):
    if epoch < 50:
        return 1.0
    elif epoch < 75:
        return 0.1
    else:
        return 0.01

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Training loop
num_epochs = 300
patience = 5
best_val_loss = float('inf')
patience_counter = 0

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
            probs = torch.softmax(output, dim=1) 
            predictions = torch.argmax(probs, dim=1)
            all_targets.extend(torch.argmax(y_batch, dim=1).cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    val_loss /= len(val_loader)
    val_accuracy = (np.array(all_predictions) == np.array(all_targets)).mean()

    # Calculate metrics
    precision = precision_score(all_targets, all_predictions, average='macro')
    recall = recall_score(all_targets, all_predictions, average='macro')
    auc = roc_auc_score(to_categorical(all_targets), all_probs, multi_class='ovr')

    print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
          f"Val Accuracy: {val_accuracy:.4f}, Precision: {precision:.4f}, "
          f"Recall: {recall:.4f}, AUC: {auc:.4f}")

    # Early stopping logic
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping triggered")
        break

    scheduler.step()

# Load best model and evaluate on test set
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

X_test_tensor = torch.FloatTensor(betti_test.values).to(device)
y_test_tensor = torch.FloatTensor(labels_test).to(device)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

all_targets = []
all_predictions = []
all_probs = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        output = model(X_batch)
        probs = torch.softmax(output, dim=1)
        predictions = torch.argmax(probs, dim=1)
        all_targets.extend(torch.argmax(y_batch, dim=1).cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# Calculate metrics
accuracy = (np.array(all_predictions) == np.array(all_targets)).mean()
precision = precision_score(all_targets, all_predictions, average='macro')
recall = recall_score(all_targets, all_predictions, average='macro')
auc = roc_auc_score(to_categorical(all_targets), all_probs, multi_class='ovr')

print(f'Test Accuracy: {accuracy:.4f}')
print(f'Test AUC: {auc:.4f}')
print(f'Test Precision: {precision:.4f}')
print(f'Test Recall: {recall:.4f}')
