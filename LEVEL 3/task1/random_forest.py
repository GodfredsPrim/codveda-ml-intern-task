# ----------------------------------------
# Level 3 Task 1: Random Forest Classifier
# ----------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv(
    "C:\\Users\\HP\\Desktop\\codveda\\level 3\\task1\\iris.csv"
)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)
print("Cross-validation accuracy:", cv_scores.mean())

# Feature importance
importances = model.feature_importances_
plt.bar(X.columns, importances)
plt.title("Feature Importance")
plt.show()

print("Random Forest Completed Successfully!")
