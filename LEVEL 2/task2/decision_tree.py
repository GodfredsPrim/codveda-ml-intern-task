# ----------------------------------------
# Level 2 Task 2: Decision Tree Classifier
# ----------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, f1_score

# ----------------------------------------
# Step 1: Load dataset
# ----------------------------------------
df = pd.read_csv(
    "C:\\Users\\HP\\Desktop\\codveda\\level 2\\task2\\iris.csv"
)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# ----------------------------------------
# Step 2: Train-test split
# ----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------------
# Step 3: Train Decision Tree
# ----------------------------------------
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# ----------------------------------------
# Step 4: Evaluate
# ----------------------------------------
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))

# ----------------------------------------
# Step 5: Visualize Tree
# ----------------------------------------
plt.figure(figsize=(12, 8))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=model.classes_,
    filled=True
)
plt.show()

print("\nDecision Tree Completed Successfully!")

