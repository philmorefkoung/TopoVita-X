import numpy as np
import timm
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import pandas as pd

num_classes = 3  # 3 for NIAID

# Convert arrays to PyTorch tensors
labels_train = torch.tensor(labels_train, dtype=torch.long)
labels_val = torch.tensor(labels_val, dtype=torch.long)
labels_test = torch.tensor(labels_test, dtype=torch.long)

# One-hot encode
labels_train = torch.nn.functional.one_hot(labels_train, num_classes=num_classes).float()
labels_val = torch.nn.functional.one_hot(labels_val, num_classes=num_classes).float()
labels_test = torch.nn.functional.one_hot(labels_test, num_classes=num_classes).float()


class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

train_dataset = CustomDataset(image_train, labels_train, transform=transform)
val_dataset = CustomDataset(image_val, labels_val, transform=transform)
test_dataset = CustomDataset(image_test, labels_test, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128)
test_loader = DataLoader(test_dataset, batch_size=128)

model = timm.create_model(
    'swinv2_tiny_window8_256', 
    pretrained=True, # pretrained for full dataset, false for limited data
    img_size=128, # 128x128 for all multiclass datasets
    patch_size=2, # change patch and window size for different image size (patch=2, window=8 for 128 size images patch=4, window=7 for 224)
    window_size=8,
    num_classes=num_classes 
)

model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

def lr_lambda(epoch):
    if epoch < 50:
        return 1.0
    elif epoch < 75:
        return 0.1
    else:
        return 0.01

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

patience = 5
best_val_loss = np.inf
patience_counter = 0

for epoch in range(100):
    model.train()
    train_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)

    print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.9f}, Val Loss: {val_loss:.9f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        patience_counter += 1
        print(f"No improvement in validation loss for {patience_counter} epochs.")

    if patience_counter >= patience:
        print("Early stopping triggered.")
        break

    scheduler.step()

model.load_state_dict(torch.load("best_model.pth"))

# eval
model.eval()
all_probs = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.cuda(), labels.cuda()
        outputs = model(images)

        probs = torch.nn.functional.softmax(outputs, dim=1)
        all_probs.extend(probs.cpu().numpy())

        true_labels = torch.argmax(labels, dim=1)
        all_labels.extend(true_labels.cpu().numpy())

all_probs = np.array(all_probs)
all_labels = np.array(all_labels)
all_preds = np.argmax(all_probs, axis=1)

# Calculate metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='macro')
recall = recall_score(all_labels, all_preds, average='macro')
auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test AUC: {auc:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
