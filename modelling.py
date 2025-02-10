#scale features
# use dropout to prevent overfitting
# use more hidden layers

import data_processing  # Imports the script and runs it
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn  # Neural network layers and loss functions
import torch.optim as optim  # Optimizers (e.g., Adam, SGD)
import torch.utils.data as data  # Data loading utilities (for batching)
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

dfstroke = data_processing.dfstroke  # Access the processed DataFrame

# Because ever_married, age and work_children were substantially intercorrelated (r > .5), I'm making a new feature, combining them with PCA

# Select relevant columns
features = dfstroke[["ever_married", "age", "work_children"]]

# Standardize the data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply PCA to reduce to 1 component
pca = PCA(n_components=1)
dfstroke["combined_feature"] = pca.fit_transform(features_scaled)

# Drop original columns if desired
dfstroke = dfstroke.drop(columns=["ever_married", "age", "work_children"])

# Check explained variance ratio
print(f"Explained variance by the component: {pca.explained_variance_ratio_[0]:.2f}")

# Define features and target
X = dfstroke['gender', 'hypertension', 'heart_disease', 'Residence_type',
       'avg_glucose_level', 'bmi', 'stroke', 'work_Govt_job',
       'work_Never_worked', 'work_Private', 'work_Self-employed',
       'smoke_Unknown', 'smoke_formerly smoked', 'smoke_never smoked',
       'smoke_smokes', 'combined_feature']  # Features
y = dfstroke["stroke"]  # Binary target (0 or 1)

# Split the data into training and testing parts
# Need to define 'data' and 'target' in the code below based on storage of X data and y 'acuity'
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

print(f"Dataset loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples.")
print(data.info)

print("Class distribution, to check for imbalance:", y_train.value_counts())

# Specify architecture of Multi-Layer Perceptron (MLP) neural network 
"""
The construction of the MLP is achieved by defining a class with functions.
The functions take the various parameters so they can be modified later as needed.
"""
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation="relu", 
                 weight_init="he", dropout_rate=0.2, batch_norm=True):
        super(MLP, self).__init__()
        
        layers = []  # List to hold all layers

        prev_size = input_size # Ensures that each layer connects correctly to the next one by having same size output/input
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))

            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            """
            BatchNorm from Troch normalises the activation function at each layer to have a mean of 0 and sd of 1.
            This prevents extreme activations from causing instability, and speeds up learning.   
            """
            if activation.lower() == "relu":
                layers.append(nn.ReLU())
            elif activation.lower() == "sigmoid":
                layers.append(nn.Sigmoid())
            elif activation.lower() == "tanh":
                layers.append(nn.Tanh())        
            """
            The "Activation Function", which allows the MLP to model non-linearity.
            Which to choose depends on the type of classification problem
            ReLu is usually used for hidden layers,
            Softmax for multi-class classification output layer, and
            Sigmoid for binary classification output layer.
            """
            ## decide whether or not to apply "regularisation"
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            """
            These parameters add 'penalties' during learning to reduce the risk of over-fitting
            """
            prev_size = hidden_size  # Update layer size for next iteration 

        ## output layer
        layers.append(nn.Linear(prev_size, output_size))
        """
        Output layer has k neurons if using softmax activation to predict levels of acuity,
        or 1 neuron if regression with no activation (or linear activation)
        """
        if output_size == 1:  # Binary classification
            layers.append(nn.Sigmoid())  # Ensures output is between 0 and 1
        elif output_size > 1:  # Multi-class classification
            pass  # Softmax is applied automatically by CrossEntropyLoss
        # Combine all layers
        self.model = nn.Sequential(*layers)
        ## weight initialization: choose type
        self._initialize_weights(weight_init)

    def _initialize_weights(self, method):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                if method.lower() == "xavier":
                    nn.init.xavier_uniform_(layer.weight)
                elif method.lower() == "he":
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)  # Initialize bias to zero

    def forward(self, x):
        if x.dim() == 1:  # If input is 1D, reshape it
            x = x.unsqueeze(0)  # Add batch dimension
        return self.model(x)
    
# Defining an MLP using the MLP class
"""
This is where the particular iteration of the MLP is defined (with the loss function )
Several variables are used to store the input parameters.
These are then fed into the MLP() class to initialise the network.
"""
# The number of input layers should match the number of features in the dataset
input_size = X_train.shape[1]   # Choose values
## The number of hidden layers
hidden_sizes = [input_size * 2, input_size]  # Or can specify exact values
"""
Likely to have just one or two hidden layers to start with.
Add more if underfitting.
To decide on the number of neurons per layer: common heuristic is 1x to 2x the input size and refine based on performance.
"""
output_size = len(set(y_train))  # Calculates number of unique values in outcome variable
activation = activation = "softmax" if output_size > 1 else "sigmoid" # Update depending on problem type
dropout_rate = 0.2
## decide whether to apply 'batch normalisation' to speed up training
batch_norm = True
weight_init = "he"
model = MLP(input_size, hidden_sizes, output_size, activation, weight_init, dropout_rate, batch_norm)
print("Here is a summary of the model build:\n", model)

# Loss and optimisation
"""
We have defined the achitecture of the model. We need to define how it learns, which is determined by loss and optimisation.
The loss function defines how the model measures "error".
If 'acuity' classes are balanced, use Cross-Entropy Loss. If imbalanced classes, use Focal Loss.
The optimiser defines how the model updates weights as it learns.
"""
class FocalLoss(nn.Module):
    """
    Implementation of Focal Loss for handling class imbalance.
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Weighting factor for imbalanced classes
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)  # Compute standard CE loss
        pt = torch.exp(-ce_loss)  # Compute pt (probability of correct class)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss  # Apply focal loss formula

        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

# Decide loss function based on class balance
def get_loss_function(use_focal_loss=False):
    if use_focal_loss:
        return FocalLoss(alpha=1, gamma=2)  # Focal Loss for imbalanced classes
    else:
        return nn.CrossEntropyLoss()  # Standard Cross-Entropy Loss

# Example usage:
use_focal_loss = False  # Change to True if dealing with class imbalance
loss_function = get_loss_function(use_focal_loss)

## Optimizer: Standard choice is Adam. 
learning_rate = 0.001
"""
Start with a learning rate of 0.001.
Increase it to 0.01 if learning is too slow.
If loss is unstable, decrease to 0.0001.
When the model runs, 'print("Loss: {avg_loss:.4f}")' (found further down) will show the loss rate as the model runs.
"""
# Example model for optimizer setup
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
def adjust_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

print("Loss function:", loss_function)
print("Optimizer:", optimizer)

# Training: iterate over epochs to train the model with early stopping
def train_model(model, train_loader, val_loader, loss_function, optimizer, 
                num_epochs=50, patience=5, min_delta=0.001, device="cpu"):
    """
    Trains the MLP model with an early stopping mechanism.

    Parameters:
    - model: The neural network (MLP)
    - train_loader: DataLoader for training data
    - val_loader: DataLoader for validation data (used for early stopping)
    - loss_function: The chosen loss function (CrossEntropyLoss, FocalLoss, etc.)
    - optimizer: The optimizer (Adam, SGD, etc.)
    - num_epochs: Maximum number of training epochs
    - patience: Number of epochs to wait for improvement before stopping
    - min_delta: Minimum improvement in validation loss required
    - device: "cpu" or "cuda" for GPU acceleration
    """
    model.to(device)  # Move model to selected device
    model.train()  # Set model to training mode
    
    best_val_loss = float("inf")  # Track best validation loss
    patience_counter = 0  # Counter for early stopping

    for epoch in range(num_epochs):
        total_loss = 0.0  # Track total loss per epoch
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)  # Move data to device
            
            optimizer.zero_grad()  # Reset gradients
            outputs = model(inputs)  # Forward pass
            loss = loss_function(outputs, targets)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            
            total_loss += loss.item()  # Accumulate loss
        
        avg_train_loss = total_loss / len(train_loader)  # Compute average training loss
        val_loss = evaluate_validation_loss(model, val_loader, loss_function, device)  # Compute validation loss

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Early Stopping Check
        if val_loss < (best_val_loss - min_delta):
            best_val_loss = val_loss  # Update best validation loss
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1  # Increment patience counter
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break  # Stop training if no improvement

# Function to compute validation loss
def evaluate_validation_loss(model, val_loader, loss_function, device="cpu"):
    """
    Computes validation loss without updating model weights.
    
    Parameters:
    - model: The trained neural network (MLP)
    - val_loader: DataLoader for validation data
    - loss_function: Loss function (CrossEntropyLoss, FocalLoss, etc.)
    - device: "cpu" or "cuda" for GPU acceleration

    Returns:
    - Average validation loss
    """
    model.eval()  # Set to evaluation mode
    total_loss = 0.0

    with torch.no_grad():  # Disable gradient computation
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            total_loss += loss.item()
    
    model.train()  # Switch back to training mode
    return total_loss / len(val_loader)  # Compute average validation loss

# Example usage:
# Assume train_loader and val_loader are already defined
device = "cuda" if torch.cuda.is_available() else "cpu"
num_epochs = 50  # Max epochs
patience = 5  # Stop if no improvement in 5 epochs
min_delta = 0.001  # Minimum improvement in loss required
train_model(model, train_loader, val_loader, loss_function, optimizer, num_epochs, patience, min_delta, device)

# Test: evaluate model performance on test data
def evaluate_model(model, test_loader, loss_function, device="cpu"):
    """
    Evaluates the trained MLP model on the test/validation dataset.
    
    Parameters:
    - model: The trained neural network (MLP)
    - test_loader: DataLoader for test/validation data
    - loss_function: The chosen loss function (CrossEntropyLoss, FocalLoss, etc.)
    - device: "cpu" or "cuda" for GPU acceleration
    """
    model.to(device)  # Move model to device
    model.eval()  # Set model to evaluation mode (disables dropout/batchnorm)

    total_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():  # Disable gradient computation for efficiency
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)  # Move data to device
            
            outputs = model(inputs)  # Forward pass
            loss = loss_function(outputs, targets)  # Compute loss
            total_loss += loss.item()

            # Compute accuracy (for classification)
            _, predicted = torch.max(outputs, 1)  # Get class with highest probability
            correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)

    avg_loss = total_loss / len(test_loader)  # Compute average loss
    accuracy = correct / total_samples * 100  # Compute accuracy percentage

    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

# Example usage:
evaluate_model(model, test_loader, loss_function, device)

# Save model 
torch.save(model.state_dict(), "mlp_model.pth")

# Load the model
model.load_state_dict(torch.load("mlp_model.pth"))
model.eval()  # Set to evaluation mode