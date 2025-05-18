import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import requests
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

# Set your team token here (obtain this by registering your team)
TOKEN = "13602610"

# Model setup
from torchvision.models import resnet18

# Preprocessing constants
mean = [0.2980, 0.2962, 0.2987]
std = [0.2886, 0.2875, 0.2889]

# Define transformation for images that are already tensors
transform = transforms.Normalize(mean=mean, std=std)

# Load model
model = resnet18(pretrained=False)
model.fc = nn.Linear(512, 44)

print("Loading model...")
try:
    # Suppress warnings and force load with weights_only=False
    import warnings
    warnings.filterwarnings("ignore")
    ckpt = torch.load("./01_MIA.pt", map_location="cpu", weights_only=False)
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

# Load the private dataset
print("Loading private dataset...")
try:
    # Suppress warnings and force load with weights_only=False
    data = torch.load("./priv_out.pt", map_location="cpu", weights_only=False)
    print(f"Private dataset loaded successfully! {len(data)} samples")
except Exception as e:
    print(f"Error loading private data: {e}")
    exit(1)

# Try to load public dataset for training the attack model
print("Loading public dataset...")
try:
    # Suppress warnings and force load with weights_only=False
    pub_data = torch.load("./pub.pt", map_location="cpu", weights_only=False)
    print(f"Public dataset loaded successfully! {len(pub_data)} samples")
except Exception as e:
    print(f"Error loading public data: {e}")
    exit(1)

# Implement the membership inference attack
def compute_membership_score(model, img_tensor, label):
    """
    Compute a membership score for a single data point.
    Higher scores indicate higher likelihood of being a member of the training set.
    """
    with torch.no_grad():
        # Get model logits
        logits = model(img_tensor.unsqueeze(0))
        
        # Compute confidence score for correct class
        softmax = torch.nn.functional.softmax(logits, dim=1)
        correct_class_confidence = softmax[0, label].item()
        
        # Compute entropy of prediction
        entropy = -torch.sum(softmax * torch.log(softmax + 1e-10)).item()
        
        # Compute loss value for this sample
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([label])).item()
        
        # Avoid division by zero
        if loss < 1e-6:
            loss = 1e-6  # Set a small constant instead of zero
        
        # Return score based on confidence and entropy
        # Higher confidence and lower entropy/loss suggests membership
        score = correct_class_confidence * (1.0 - entropy/5.0) * (1.0/loss)
        
        # Normalize to [0,1] range
        score = max(0.0, min(1.0, score))
        
        return score

# Process dataset and compute membership scores
scores = []
print("Processing private dataset...")
for i in range(len(data)):
    try:
        # Extract data properly based on dataset format
        if hasattr(data, 'ids') and hasattr(data, 'imgs') and hasattr(data, 'labels'):
            # Handle MembershipDataset format
            id_ = data.ids[i]
            img = data.imgs[i]
            label = data.labels[i]
        else:
            # Handle tuple format (id, img, label)
            batch = data[i]
            if len(batch) >= 3:  # Make sure we have at least 3 elements
                id_ = batch[0]
                img = batch[1]
                label = batch[2]
            else:
                print(f"Unexpected data format at index {i}")
                continue
        
        # Skip ToTensor() since the image is already a tensor
        # Just apply normalization
        img_tensor = img
        if transform is not None:
            # Add batch dimension if needed
            if len(img_tensor.shape) == 3:
                img_tensor = transform(img_tensor)
        
        # Compute membership score
        score = compute_membership_score(model, img_tensor, label)
        scores.append((id_, score))
        
        # Progress indicator
        if i % 1000 == 0:
            print(f"Processed {i}/{len(data)} samples")
            
    except Exception as e:
        print(f"Error processing sample {i}: {e}")
        # Skip this sample if we can't process it

# Prepare submission
df = pd.DataFrame({
    "ids": [s[0] for s in scores],
    "score": [s[1] for s in scores],
})

# Save to CSV
df.to_csv("submission.csv", index=None)

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