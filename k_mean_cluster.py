import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('corrected_dataset.csv')  # Replace 'tr1.csv' with your dataset path
categorical_columns = data.select_dtypes(include=['object']).columns

# Apply one-hot encoding to categorical columns
data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
# Preprocess data: Standardize the features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_encoded)

# Set number of clusters
n_clusters = 3  # Adjust as needed

# Perform K-Means clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
labels = kmeans.fit_predict(data_scaled)
centroids = kmeans.cluster_centers_

# Calculate Silhouette Score
silhouette_avg = silhouette_score(data_scaled, labels)

# Output results
print(f"Number of Clusters: {n_clusters}")
print(f"Silhouette Score: {silhouette_avg:.4f}")
print(f"Cluster Centroids:\n{centroids}")

# Plot clusters (assuming 2D or reduced to 2D via PCA for visualization)
plt.figure(figsize=(10, 6))
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=labels, cmap='viridis', s=50)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title(f"K-Means Clustering (k={n_clusters}), Silhouette Score={silhouette_avg:.4f}", fontsize=14)
plt.xlabel('Feature 1 (Standardized)', fontsize=12)
plt.ylabel('Feature 2 (Standardized)', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
