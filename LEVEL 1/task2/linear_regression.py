# ----------------------------------------
# Level 1 Task 2: Simple Linear Regression
# ----------------------------------------

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ----------------------------------------
# Step 1: Load dataset
# ----------------------------------------
df = pd.read_csv(
    "C:\\Users\\HP\\Desktop\\codveda\\level 1\\task2\\stock_prices.csv"
)

# ----------------------------------------
# Step 2: Keep only numeric columns
# (VERY IMPORTANT FIX)
# ----------------------------------------
df = df.select_dtypes(include=[np.number])

# Handle missing values
df.fillna(df.mean(), inplace=True)

print("Numeric columns used:")
print(df.columns)

# ----------------------------------------
# Step 3: Select ONE feature & target
# (Simple Linear Regression)
# ----------------------------------------
X = df.iloc[:, [0]]   # First numeric feature
y = df.iloc[:, -1]    # Target (stock price)

# ----------------------------------------
# Step 4: Train-test split
# ----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------------
# Step 5: Train model
# ----------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ----------------------------------------
# Step 6: Predict
# ----------------------------------------
y_pred = model.predict(X_test)

# ----------------------------------------
# Step 7: Evaluate
# ----------------------------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-squared Score:", r2)

print("\nCoefficient:", model.coef_[0])
print("Intercept:", model.intercept_)

print("\nSimple Linear Regression Completed Successfully!")

