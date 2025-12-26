# ----------------------------------------
# Level 3 Task 2: SVM Classification
# ----------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score

# Load dataset
df = pd.read_csv(
    "C:\\Users\\HP\\Desktop\\codveda\\level 3\\task2\\iris.csv"
)

# Convert to binary classification
df = df[df.iloc[:, -1] != 'Iris-setosa']

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Encode labels
y = (y == 'Iris-virginica').astype(int)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM (RBF Kernel)
model = SVC(kernel='rbf', probability=True)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC Score:", roc_auc_score(y_test, y_prob))

print("SVM Completed Successfully!")

