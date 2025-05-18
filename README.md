# TML 2025 - Task 1: Membership Inference Attack
## Team 34

## Overview

This repository contains our solution for Task 1 of the TML 2025 course, which focuses on implementing membership inference attacks against a trained ResNet-18 model. The goal is to determine whether particular data points were used to train a machine learning model, thereby exposing potential privacy vulnerabilities.

## Problem Description

Membership inference attacks aim to determine if a specific data sample was part of a model's training set. This is a privacy concern because it can potentially leak information about the training data. In this task:

1. We are given a ResNet-18 model trained on an undisclosed dataset
2. We have access to a public dataset with known membership labels (member=1, non-member=0)
3. We need to predict membership scores for a private dataset with unknown membership labels

## Our Approach

We implemented two different approaches to solve this task:

### 1. Basic Approach (`membership_inference.py`)

Our basic approach relies on heuristic features that typically differ between training and non-training samples:

- **Confidence Score**: Models tend to output higher confidence predictions for samples they were trained on
- **Entropy**: Predictions on training samples typically have lower entropy (more certainty)
- **Loss Value**: Training samples usually have lower loss values

The membership score is calculated as:
```python
score = correct_class_confidence * (1.0 - entropy/5.0) * (1.0/loss)
```

This formula gives higher scores to samples with:
- Higher confidence for the correct class
- Lower prediction entropy
- Lower loss values

### 2. Advanced Approach (`membership_inference_advanced.py`)

Our advanced approach trains a supervised attack model on the public dataset with known membership information:

1. **Feature Extraction**:
   - Extract various features from the target model's behavior on each sample:
     - Class probabilities (all classes)
     - Loss value for the sample
     - Prediction correctness
     - Maximum probability
     - Prediction entropy
     - Margin between top two class probabilities

2. **Attack Model Training**:
   - We train a 3-layer neural network on these features to predict membership
   - The architecture consists of:
     - Input layer: Feature dimension (varies based on number of classes)
     - Hidden layers: 64 and 32 neurons with ReLU activation and dropout
     - Output layer: Single neuron with sigmoid activation for binary classification
   - We train for 20 epochs using BCELoss and Adam optimizer

3. **Prediction**:
   - Apply the same feature extraction to the private dataset
   - Use the trained attack model to predict membership scores

## Implementation Details

### Data Handling

We carefully processed the input data to handle:
- Tensor format conversion
- Normalization using the provided mean and std values
- Error handling for edge cases (e.g., division by zero)

### Model Architecture

The attack model architecture in our advanced approach:
```python
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
```

### Key Files

- `membership_inference.py`: Contains the basic approach implementation
- `membership_inference_advanced.py`: Contains the advanced approach with attack model training
- `submission.csv`: Our final submission file with membership scores

## Results and Analysis

We found that the advanced approach generally outperforms the basic approach due to its ability to learn complex patterns from the provided public dataset with ground truth membership information.

The training curves of our attack model show convergence with validation loss stabilizing around 0.62, indicating a good balance between overfitting and underfitting.

Some interesting observations:
- Certain classes showed stronger membership signals than others
- The margin between the top class and second-highest class proved to be a particularly informative feature
- There's a clear correlation between loss values and membership status

## Usage Instructions

1. Clone this repository
2. Set up the environment:
   ```
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install torch torchvision numpy pandas scikit-learn requests
   ```
3. Run either approach:
   ```
   python membership_inference.py  # for basic approach
   python membership_inference_advanced.py  # for advanced approach
   ```

## Conclusion

Our implementation demonstrates how machine learning models can inadvertently leak information about their training data. The advanced approach particularly shows how this privacy vulnerability can be exploited systematically with relatively simple techniques. This underscores the importance of privacy-preserving machine learning methods, especially for applications involving sensitive data. 