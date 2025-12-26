# ----------------------------------------
# Level 1 Task 1: Data Preprocessing
# ----------------------------------------

# Import libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ----------------------------------------
# Step 1: Load the dataset
# ----------------------------------------
df = pd.read_csv("C:\\Users\\HP\\Desktop\\codveda\\level 1\\task1\\stock_prices.csv")

print("First 5 rows of the dataset:")
print(df.head())

# ----------------------------------------
# Step 2: Understand the dataset
# ----------------------------------------
print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# ----------------------------------------
# Step 3: Handle missing values
# Fill numerical columns with mean
# ----------------------------------------
df.fillna(df.mean(numeric_only=True), inplace=True)

print("\nMissing values after handling:")
print(df.isnull().sum())

# ----------------------------------------
# Step 4: Encode categorical variables (if any)
# ----------------------------------------
df = pd.get_dummies(df, drop_first=True)

print("\nDataset after encoding categorical variables:")
print(df.head())

# ----------------------------------------
# Step 5: Separate features and target
# (Assuming last column is the target)
# ----------------------------------------
X = df.iloc[:, :-1]   # Features
y = df.iloc[:, -1]    # Target

# ----------------------------------------
# Step 6: Normalize the features
# ----------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------------------
# Step 7: Split the dataset
# ----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42
)

print("\nTraining and Testing Data Shapes:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

print("\n✅ Data Preprocessing Completed Successfully!")

