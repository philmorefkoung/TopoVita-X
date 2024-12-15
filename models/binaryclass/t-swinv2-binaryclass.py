import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

class HybridModel(nn.Module):
    def __init__(self, feature_dim=400, num_classes=1):  
        super().__init__()
        
        # Vision Transformer branch
        self.vision_model = timm.create_model(
            'swinv2_tiny_window8_256',
            pretrained=True, # pretrained for full datasets only
            img_size=128, # 128x128 for all datasets, 224x224 for ALL-IDB2
            patch_size=2, # 2 for 128x128 images, 4 for 224x224
            window_size=8, # 8 for 128x128 images, 7 for 224x224
            num_classes=0
        )
        
        vision_output_dim = self.vision_model.num_features
        
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.vision_projection = nn.Sequential(
            nn.Linear(vision_output_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),
            
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 1)
        )

    def forward(self, images, features):
        vision_features = self.vision_model.forward_features(images)
        
        B, H, W, C = vision_features.shape
        vision_features = vision_features.reshape(B, H*W, C)
        vision_features = vision_features.mean(dim=1)
        
        vision_features = self.vision_projection(vision_features)
        mlp_features = self.mlp(features)
        
        combined_features = torch.cat([vision_features, mlp_features], dim=1)
        
        output = self.fusion(combined_features)
        return output

class HybridDataset(Dataset):
    def __init__(self, images, features, labels, transform=None):
        self.images = images
        self.features = features
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        feature = self.features[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.float32) 

def load_presplit_data(train_npz, val_npz, test_npz, 
                      train_csv, val_csv, test_csv, 
                      batch_size=128):

    train_data = np.load(train_npz)
    val_data = np.load(val_npz)
    test_data = np.load(test_npz)
    
    train_features = pd.read_csv(train_csv).values
    val_features = pd.read_csv(val_csv).values
    test_features = pd.read_csv(test_csv).values
    
    train_images = train_data['images']
    train_labels = train_data['labels']
    val_images = val_data['images']
    val_labels = val_data['labels']
    test_images = test_data['images']
    test_labels = test_data['labels']
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    
    train_dataset = HybridDataset(train_images, train_features, train_labels, transform=transform)
    val_dataset = HybridDataset(val_images, val_features, val_labels, transform=transform)
    test_dataset = HybridDataset(test_images, test_features, test_labels, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def train_hybrid_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    num_epochs=100,
    learning_rate=1e-4,
    patience=5,
    device='cuda'
):
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss() 
    
    def lr_lambda(epoch):
        if epoch < 50:
            return 1.0
        elif epoch < 75:
            return 0.1
        else:
            return 0.01
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': []
    }
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for images, features, labels in train_loader:
            images = images.to(device)
            features = features.to(device)
            labels = labels.to(device).view(-1, 1) 
            
            optimizer.zero_grad()
            outputs = model(images, features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, features, labels in val_loader:
                images = images.to(device)
                features = features.to(device)
                labels = labels.to(device).view(-1, 1)
                
                outputs = model(images, features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rates'].append(scheduler.get_last_lr()[0])
        
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        print("-" * 50)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, "best_hybrid_model.pth")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs")
            
        if patience_counter >= patience:
            print("Early stopping triggered")
            break
        
        scheduler.step()
    
    model.load_state_dict(torch.load("best_hybrid_model.pth"))
    
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, features, labels in test_loader:
            images = images.to(device)
            features = features.to(device)
            
            outputs = model(images, features)
            probs = torch.sigmoid(outputs).squeeze()  
            preds = (probs > 0.5).float()
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds),
        'recall': recall_score(all_labels, all_preds),
        'auc': roc_auc_score(all_labels, all_probs)
    }
    
    return model, history, metrics

if __name__ == "__main__":
    model = HybridModel(feature_dim=400, num_classes=1)  
    
    train_loader, val_loader, test_loader = load_presplit_data(
        train_npz='train_npz_name.npz',
        val_npz='val_npz_name.npz',
        test_npz='test_npz_name.npz',

        train_csv='train_csv_name.csv',
        val_csv='val_csv_name.csv',
        test_csv='test_csv_name.csv',

        batch_size=128
    )
    
    trained_model, history, metrics = train_hybrid_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=300,
        learning_rate=1e-4,
        patience=5,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    # Print final metrics
    print("\nFinal Test Metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name.capitalize()}: {value:}")