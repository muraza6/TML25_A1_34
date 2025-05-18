TML 2025 - Task 1: Membership Inference Attack
Team 34

OVERVIEW

This repository contains our solution for Task 1 of the TML 2025 course, which focuses on implementing membership inference attacks against a trained ResNet-18 model. The goal is to determine whether particular data points were used to train a machine learning model, thereby exposing potential privacy vulnerabilities.

PROBLEM DESCRIPTION

Membership inference attacks aim to determine if a specific data sample was part of a model's training set. This is a privacy concern because it can potentially leak information about the training data. In this task:

1. We are given a ResNet-18 model trained on an undisclosed dataset
2. We have access to a public dataset with known membership labels (member=1, non-member=0)
3. We need to predict membership scores for a private dataset with unknown membership labels

OUR APPROACH

Our approach relies on heuristic features that typically differ between training and non-training samples:

- Confidence Score: Models tend to output higher confidence predictions for samples they were trained on
- Entropy: Predictions on training samples typically have lower entropy (more certainty)
- Loss Value: Training samples usually have lower loss values

The membership score is calculated as:
```python
score = correct_class_confidence * (1.0 - entropy/5.0) * (1.0/loss)
```

This formula gives higher scores to samples with:
- Higher confidence for the correct class
- Lower prediction entropy
- Lower loss values

IMPLEMENTATION DETAILS

Data Handling

We carefully processed the input data to handle:
- Tensor format conversion
- Normalization using the provided mean and std values
- Error handling for edge cases (e.g., division by zero)

Key Files

- `membership_inference.py`: Contains our membership inference implementation
- `submission.csv`: Our final submission file with membership scores

RESULTS AND ANALYSIS

Our solution effectively identifies membership patterns in the model's behavior.

Some interesting observations:
- There's a clear correlation between loss values and membership status
- The combination of confidence, entropy, and loss provides a robust signal for membership inference
- Properly handling the normalization and tensor conversions proved crucial for accurate results

USAGE INSTRUCTIONS

1. Clone this repository
2. Set up the environment:
   ```
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install torch torchvision numpy pandas scikit-learn requests
   ```
3. Run the implementation:
   ```
   python membership_inference.py
   ```

CONCLUSION

Our implementation demonstrates how machine learning models can inadvertently leak information about their training data. This underscores the importance of privacy-preserving machine learning methods, especially for applications involving sensitive data. 