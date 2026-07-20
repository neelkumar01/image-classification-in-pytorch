## MNIST Handwritten Digit Classification using PyTorch 

### Project Overview

This project implements a fully configurable Multi Layer Perceptron (MLP) from scratch using PyTorch to classify handwritten digits from the MNIST dataset.

Rather than building a single neural network, the project focuses on creating a complete deep learning training pipeline that allows experimenting with different network architectures, activation functions, regularization techniques, and optimization strategies

The objective was not only to achieve high classification accuracy but also to understand every stage involved in training deep neural networks—from data preprocessing to model evaluation, checkpointing, and hyperparameter experimentation

### Dataset

The project uses the MNIST Handwritten Digit Dataset, one of the most widely used benchmark datasets for image classification

### Architecture
```
                 MNIST Dataset
                       │
                       ▼
              Data Preprocessing
        (Tensor + Normalization)
                       │
                       ▼
          Train / Validation Split
                       │
                       ▼
                DataLoaders
                       │
                       ▼
           Configurable MLP Model
                       │
         ┌─────────────┴─────────────┐
         │                           │
   Forward Pass               Loss Function
         │                    CrossEntropy
         │                           │
         └─────────────┬─────────────┘
                       ▼
                Adam Optimizer
                       │
          ReduceLROnPlateau Scheduler
                       │
               Early Stopping
                       │
                       ▼
          Best Model Checkpoint Saved
                       │
                       ▼
         Evaluation on Test Dataset
                       │
                       ▼
          Experiment Comparison

```

### Best Practices Demonstrated

- Data normalization using dataset statistics (mean=0.1307, std=0.3081)
- Train/Val/Test split for unbiased evaluation
- CrossEntropyLoss with raw logits (no softmax in model)
- Adam optimizer with weight decay for regularization
- Learning rate scheduling (ReduceLROnPlateau)
- Early stopping to prevent overfitting
- Dropout for regularization
- Model checkpointing (save best validation model)
