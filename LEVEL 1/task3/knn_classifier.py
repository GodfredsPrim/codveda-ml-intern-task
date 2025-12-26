# ----------------------------------------
# Level 1 Task 3: K-Nearest Neighbors (KNN)
# ----------------------------------------

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ----------------------------------------
# Step 1: Load dataset
# ----------------------------------------
df = pd.read_csv(
    "C:\\Users\\HP\\Desktop\\codveda\\level 1\\task3\\iris.csv"
)

print("First 5 rows:")
print(df.head())

# ----------------------------------------
# Step 2: Separate features and target
# ----------------------------------------
X = df.iloc[:, :-1]   # Features
y = df.iloc[:, -1]    # Target (species)

# ----------------------------------------
# Step 3: Train-test split
# ----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------------
# Step 4: Feature scaling
# ----------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----------------------------------------
# Step 5: Train KNN model (K = 5)
# ----------------------------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# ----------------------------------------
# Step 6: Predictions
# ----------------------------------------
y_pred = knn.predict(X_test)

# ----------------------------------------
# Step 7: Evaluation
# ----------------------------------------
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("\nConfusion Matrix:\n", cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nKNN Classifier Completed Successfully!")

