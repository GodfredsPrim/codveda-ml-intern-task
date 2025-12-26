# ----------------------------------------
# Level 2 Task 1: Logistic Regression
# ----------------------------------------

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# ----------------------------------------
# Step 1: Load dataset
# ----------------------------------------
df = pd.read_csv(
    "C:\\Users\\HP\\Desktop\\codveda\\level 2\\task1\\stock_prices.csv"
)

# Keep numeric columns only
df = df.select_dtypes(include=[np.number])
df.fillna(df.mean(), inplace=True)

# ----------------------------------------
# Step 2: Create binary target
# 1 = price above mean, 0 = below mean
# ----------------------------------------
df['target'] = (df['close'] > df['close'].mean()).astype(int)

X = df[['open', 'high', 'low', 'volume']]
y = df['target']

# ----------------------------------------
# Step 3: Split data
# ----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------------
# Step 4: Scale features
# ----------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----------------------------------------
# Step 5: Train Logistic Regression
# ----------------------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# ----------------------------------------
# Step 6: Evaluate
# ----------------------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_prob))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nLogistic Regression Completed Successfully!")

