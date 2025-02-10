import data_processing  # Imports the script and runs it
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dfstroke = data_processing.dfstroke  # Access the processed DataFrame

# Compute correlation matrix
corr_matrix = dfstroke.corr()

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Between Features")
plt.show()