# Neural Networks

This project implements **one-layer and two-layer neural networks** for **wine quality regression** and a **CNN for MNIST digit classification**, trained with stochastic gradient descent (SGD). It includes feature normalization, loss evaluation, and visualization utilities.

## Features

- **One-Layer NN**: Single-layer linear network with MSE loss for wine quality prediction.  
- **Two-Layer NN**: Hidden layer with 32 units and sigmoid activation.  
- **CNN**: Convolutional network for MNIST 8×8 digit classification with cross-entropy loss.  
- **Stochastic Gradient Descent (SGD)**: Optimizes weights and biases.  
- **Loss Metrics**: Computes average MSE loss (wine) or cross-entropy loss (MNIST).  
- **Accuracy Metrics**: Calculates classification accuracy for MNIST.  
- **Visualization**: Training loss, accuracy, misclassified images, and confusion matrix.  
- **Feature Normalization**: Automatically normalizes input features for wine dataset.  

## How to Run

The main entry point is `main.py`. Running it will train and evaluate the models:

```bash
python main.py
```

The script will:
- Load the wine.txt dataset for regression and digits.csv for MNIST classification.
- Normalize features and split into training/testing sets.
- Train and evaluate One-Layer NN, Two-Layer NN, and CNN models.
- Print average training and testing losses and accuracies.
- Optionally visualize training metrics, predictions, and confusion matrix.

## Project Structure
```text
.
├── data/                  # Folder containing dataset files (wine.txt, digits.csv)
├── main.py                # Main script to train and test the models
├── models.py              # Implements OneLayerNN, TwoLayerNN, CNN, and utility functions
├── utils.py               # Data loaders and visualization functions
└── README.md              # Project documentation
