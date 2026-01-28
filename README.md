Skin Cancer Detection: Custom CNN Model
Project Overview
Project Title: A Comparative Analysis of Deep Learning Architectures for Automated Skin Cancer Detection
Student: Thouheid Ahmed
SRN: 21084799
Supervisor: Jan Kim

This is Custom CNN - a custom Convolutional Neural Network (CNN) built from scratch for binary classification of skin cancer images (benign vs malignant).

Model Purpose
The custom CNN serves as a baseline model to:

Demonstrate fundamental understanding of CNN architecture

Compare against pre-trained models (VGG16, ResNet50)

Establish performance benchmarks for the skin cancer detection task

INPUT: [64×64×3] RGB Images
    ↓
CONV2D: 32 filters (3×3), ReLU activation
    ↓
MAX POOLING: (2×2) → [32×32×32]
    ↓
CONV2D: 64 filters (3×3), ReLU activation
    ↓
MAX POOLING: (2×2) → [16×16×64]
    ↓
CONV2D: 64 filters (3×3), ReLU activation
    ↓
FLATTEN: → 16,384 features
    ↓
DENSE: 64 neurons, ReLU activation
    ↓
OUTPUT: 1 neuron, Sigmoid activation (Binary classification)

(Model Size 5MB)

Key Features
1. Data Pipeline
Image Size: 64×64 pixels (optimized for training speed)

Data Augmentation:

Rotation (±20°)

Horizontal/Vertical shifting (±20%)

Horizontal flipping

Data Split:

Training: 80% of training data

Validation: 20% of training data

Testing: Separate test set

2. Training Configuration
Optimizer: Adam (adaptive learning rate)

Loss Function: Binary Crossentropy

Batch Size: 32 images

Epochs: 10

Metrics Tracked:

Accuracy

Precision

Recall

3. Evaluation Metrics
Accuracy: Overall correctness

Precision: Correct positive predictions (avoid false alarms)

Recall: Detection of actual positives (find all cancers)

Loss: Training convergence indicator

Training Process:
Data Loading: Images loaded from Google Drive

Preprocessing: Resize to 64×64, normalize pixel values (0-1)

Model Building: Sequential CNN construction

Training: 10 epochs with validation monitoring

Evaluation: Test set performance assessment

Visualization: Accuracy/Loss plots, sample predictions

Saving: Model saved as .h5 file

Expected Performance
Target Metrics:
Accuracy Target: ≥85%

Precision: High (minimize false positives)

Recall: High (minimize false negatives)

Training Characteristics:
Training Time: ~5-10 minutes on Google Colab GPU

Memory Usage: Low (~5 MB model size)

Inference Speed: Fast (suitable for real-time applications)

Comparative Advantages
Strengths:
Simplicity: Easy to understand and modify

Lightweight: Fast training and inference

Educational: Demonstrates CNN fundamentals

Customizable: Full control over architecture

Interpretable: Easy to debug and analyze

Limitations:
Feature Extraction: Less powerful than pre-trained models

Data Requirements: Needs more data for optimal performance

Convergence: May require careful hyperparameter tuning

Output Files
Generated Files:
Trained Model: fighter1_trained_model.h5

Visualizations:

Accuracy vs Epochs plot

Loss vs Epochs plot

Precision/Recall vs Epochs plot

Sample predictions with confidence scores

Results Interpretation
Key Observations:
Training Curves: Monitor for overfitting (validation loss increasing)

Accuracy Plateau: Indicates model has learned available patterns

Precision-Recall Tradeoff: Medical context may prioritize recall over precision

Sample Predictions: Visual verification of model performance

Next Steps
After Training:
Evaluate against test set

Compare with VGG16 and ResNet50 results

Analyze misclassified cases

Optimize hyperparameters if needed

Document findings in dissertation

Improvement Opportunities:
Increase image size to 224×224

Add more convolutional layers

Implement batch normalization

Add dropout for regularization

Experiment with different optimizers

Important Notes
Dataset Balance: Ensure balanced benign/malignant samples

Image Quality: Verify image preprocessing is consistent

Model Checkpoints: Save best model during training

Ethical Considerations: Medical AI requires careful validation

Reproducibility: Set random seeds for consistent results

Academic Context
This custom CNN serves as:

Baseline Model: For comparison with advanced architectures

Learning Tool: Demonstrates CNN fundamentals in medical imaging

Research Contribution: Provides insights into model design tradeoffs

Practical Application: Shows real-world AI implementation for healthcare

