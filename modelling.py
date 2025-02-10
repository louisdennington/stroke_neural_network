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

print(dfstroke.columns())

# Define features and target
X = data["combined_feature"]  # Features
y = data["stroke"]  # Binary target (0 or 1)
