import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import requests
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from typing import Tuple
import os

# Set your team token here (obtain this by registering your team)
TOKEN = "13602610"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model setup
from torchvision.models import resnet18

# Preprocessing constants
mean = [0.2980, 0.2962, 0.2987]
std = [0.2886, 0.2875, 0.2889]

# Define transformation for images that are already tensors
transform = transforms.Normalize(mean=mean, std=std)

# Load target model
model = resnet18(pretrained=False)
model.fc = nn.Linear(512, 44)
model = model.to(device)

print("Loading model...")
try:
    # Suppress warnings and force load with weights_only=False
    import warnings
    warnings.filterwarnings("ignore")
    ckpt = torch.load("./01_MIA.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Set model to evaluation mode
model.eval()

# Dataset classes (from template)
class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids = []
        self.imgs = []
        self.labels = []
        self.transform = transform

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int]:
        id_ = self.ids[index]
        img = self.imgs[index]
        if not self.transform is None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)

class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.membership = []

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int, int]:
        id_, img, label = super().__getitem__(index)
        return id_, img, label, self.membership[index]

def extract_features(model, dataset, batch_size=32):
    """Extract features to use for membership inference attack"""
    all_ids = []
    all_features = []
    all_labels = []
    all_membership = []
    
    # Process samples individually to avoid batch collation errors
    print("Extracting features...")
    with torch.no_grad():
        for i in range(len(dataset)):
            try:
                # Get data from dataset
                if hasattr(dataset, 'ids') and hasattr(dataset, 'imgs') and hasattr(dataset, 'labels'):
                    # Handle MembershipDataset format
                    id_ = dataset.ids[i]
                    img = dataset.imgs[i]
                    label = dataset.labels[i]
                    membership = None
                    if hasattr(dataset, 'membership') and i < len(dataset.membership):
                        membership = dataset.membership[i]
                else:
                    # Handle tuple format (id, img, label, membership?)
                    batch = dataset[i]
                    if len(batch) >= 3:  # Make sure we have at least 3 elements
                        id_ = batch[0]
                        img = batch[1]
                        label = batch[2]
                        membership = None if len(batch) < 4 else batch[3]
                    else:
                        print(f"Unexpected data format at index {i}")
                        continue
                
                # Move to device
                img = img.to(device)
                if isinstance(label, torch.Tensor):
                    label = label.to(device)
                else:
                    label = torch.tensor(label, device=device)
                
                # Apply normalization if needed (skip ToTensor as data is already tensor)
                if transform is not None:
                    img = transform(img)
                    
                # Add batch dimension
                img = img.unsqueeze(0)
                
                # Get model outputs
                outputs = model(img)
                
                # Calculate softmax probabilities
                probs = torch.nn.functional.softmax(outputs, dim=1)
                
                # Calculate loss for each sample
                losses = torch.nn.functional.cross_entropy(outputs, label.unsqueeze(0), reduction='none')
                
                # Calculate predicted labels
                _, preds = torch.max(outputs, 1)
                
                # Calculate if prediction matches true label
                correct = (preds == label).float()
                
                # Calculate max probability
                max_probs, _ = torch.max(probs, dim=1)
                
                # Calculate entropy
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
                
                # Calculate margin (difference between correct class probability and highest other class)
                onehot = torch.zeros_like(probs)
                onehot.scatter_(1, label.unsqueeze(0).unsqueeze(1), 1)
                correct_probs = torch.sum(probs * onehot, dim=1)
                
                # Get second highest probability
                sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
                second_probs = sorted_probs[:, 1]
                
                # Calculate margins
                margins = correct_probs - second_probs
                
                # Concatenate features
                features = torch.cat([
                    probs,  # All class probabilities
                    losses.unsqueeze(1),  # Loss
                    correct.unsqueeze(1),  # Correctness
                    max_probs.unsqueeze(1),  # Maximum probability
                    entropy.unsqueeze(1),  # Entropy
                    margins.unsqueeze(1),  # Margin
                ], dim=1)
                
                # Move to CPU and convert to numpy
                all_ids.append(id_)
                all_features.append(features.cpu().numpy())
                all_labels.append(label.cpu().numpy())
                if membership is not None:
                    all_membership.append(membership)
                    
                # Show progress
                if i % 1000 == 0:
                    print(f"Processed {i}/{len(dataset)} samples")
                    
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
    
    # Concatenate features and convert everything to numpy arrays
    all_features = np.vstack(all_features)
    all_ids = np.array(all_ids)
    all_labels = np.array(all_labels)
    
    # Handle case where membership is None for all samples
    if len(all_membership) == 0:
        all_membership = None
    else:
        all_membership = np.array(all_membership)
    
    return all_ids, all_features, all_labels, all_membership

# Load datasets
print("Loading public dataset...")
try:
    # Suppress warnings and force load with weights_only=False
    pub_data = torch.load("./pub.pt", map_location=device, weights_only=False)
    print(f"Public dataset loaded: {len(pub_data)} samples")
except Exception as e:
    print(f"Error loading public data: {e}")
    exit(1)

print("Loading private dataset...")
try:
    # Suppress warnings and force load with weights_only=False
    priv_data = torch.load("./priv_out.pt", map_location=device, weights_only=False)
    print(f"Private dataset loaded: {len(priv_data)} samples")
except Exception as e:
    print(f"Error loading private data: {e}")
    exit(1)

# Extract features from public dataset
print("Extracting features from public dataset...")
pub_ids, pub_features, pub_labels, pub_membership = extract_features(model, pub_data)
print(f"Public features shape: {pub_features.shape}")

# Train attack model on public dataset
print("Training attack model...")

# Define attack model
class AttackModel(nn.Module):
    def __init__(self, input_size):
        super(AttackModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x

# Split public dataset for training and validation
X_train, X_val, y_train, y_val = train_test_split(
    pub_features, pub_membership, test_size=0.2, random_state=42
)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.FloatTensor(y_val)

# Create data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor.unsqueeze(1))
val_dataset = TensorDataset(X_val_tensor, y_val_tensor.unsqueeze(1))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

# Initialize attack model
attack_model = AttackModel(X_train.shape[1]).to(device)

# Define optimizer and loss
optimizer = optim.Adam(attack_model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Training loop
n_epochs = 20
best_val_loss = float('inf')
best_model_path = "best_attack_model.pt"

for epoch in range(n_epochs):
    # Training
    attack_model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        # Forward pass
        y_pred = attack_model(X_batch)
        loss = criterion(y_pred, y_batch)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # Validation
    attack_model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = attack_model(X_batch)
            val_loss += criterion(y_pred, y_batch).item()
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(attack_model.state_dict(), best_model_path)
    
    print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")

# Load best model
attack_model.load_state_dict(torch.load(best_model_path))
attack_model.eval()

# Extract features from private dataset
print("Extracting features from private dataset...")
priv_ids, priv_features, priv_labels, _ = extract_features(model, priv_data)

# Generate membership scores for private dataset
priv_features_tensor = torch.FloatTensor(priv_features).to(device)
with torch.no_grad():
    priv_scores = attack_model(priv_features_tensor).cpu().numpy().flatten()

# Prepare submission
df = pd.DataFrame({
    "ids": priv_ids,
    "score": priv_scores,
})

# Save to CSV
df.to_csv("submission.csv", index=None)
print(f"Submission saved to submission.csv with {len(df)} samples")

# Submit to evaluation server
print("Submitting to evaluation server...")
try:
    # Use the token from the variable defined at the top
    response = requests.post(
        "http://34.122.51.94:9090/mia", 
        files={"file": open("submission.csv", "rb")}, 
        headers={"token": TOKEN}
    )
    print(response.json())
except Exception as e:
    print(f"Error submitting: {e}")
    print("Please submit manually using the CSV file")

# Clean up
if os.path.exists(best_model_path):
    os.remove(best_model_path)