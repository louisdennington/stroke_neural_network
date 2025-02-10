import pandas as pd

dfstroke = pd.read_csv(r'C:\Users\louis\OneDrive - University College London\MSc Health Data Science\0 - Personal projects\stroke_neural_network\healthcare-dataset-stroke-data.csv')
print(dfstroke.info())

# Drop ID variable
dfstroke = dfstroke.drop(columns=["id"])

# Convert 'yes'/'no' in 'ever_married' to 1/0
dfstroke['ever_married'] = dfstroke['ever_married'].map({'Yes': 1, 'No': 0})

# Convert 'Male'/'Female' in 'gender' to numeric
dfstroke['gender'] = dfstroke['gender'].map({'Male': 1, 'Female': 0})

# Check for null values by column
print("Null values per column:\n", dfstroke.isnull().sum())

# BMI is the only column with missing values (n = 201). Filling with mean. 
dfstroke['bmi'].fillna(dfstroke['bmi'].mean(), inplace=True)

# Get frequency counts for specified categorical columns
print("\nFrequency counts:")
for col in ['gender', 'hypertension', 'heart_disease']:
    print(f"{col}:\n{dfstroke[col].value_counts()}\n")

# Compute summary statistics for specified numeric columns
print("\nSummary statistics:")
print(dfstroke[['age', 'avg_glucose_level', 'bmi']].describe().loc[['min', 'max', 'mean', 'std']])

# Recoding other categorical variables

# Encoding 'work_type' with one-hot encoding
dfstroke = pd.get_dummies(dfstroke, columns=['work_type'], prefix='work')

# Encoding 'Residence_type' as binary (0 = Rural, 1 = Urban)
dfstroke['Residence_type'] = dfstroke['Residence_type'].map({'Rural': 0, 'Urban': 1})

# Encoding 'smoking_status' with one-hot encoding
dfstroke = pd.get_dummies(dfstroke, columns=['smoking_status'], prefix='smoke')

# Display the first few rows to check the transformations
print(dfstroke.head())

# Prevent execution when imported
if __name__ == "__main__":
    print("Data processing complete.")