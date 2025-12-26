# ----------------------------------------
# Level 2 Task 3: K-Means Clustering
# ----------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ----------------------------------------
# Step 1: Load dataset
# ----------------------------------------
df = pd.read_csv(
    "C:\\Users\\HP\\Desktop\\codveda\\level 2\\task3\\stock_prices.csv"
)

df = df.select_dtypes(include=[np.number])
df.fillna(df.mean(), inplace=True)

# ----------------------------------------
# Step 2: Scale data
# ----------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# ----------------------------------------
# Step 3: Elbow Method
# ----------------------------------------
wcss = []

for k in range(1, 8):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 8), wcss, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

# ----------------------------------------
# Step 4: Apply K-Means
# ----------------------------------------
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df['Cluster'] = clusters

print(df.head())
print("\nK-Means Clustering Completed Successfully!")

