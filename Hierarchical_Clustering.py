import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('corrected_dataset.csv')  # Replace 'tr1.csv' with your dataset path

# Preprocess data: Convert categorical data to numeric if needed
categorical_columns = data.select_dtypes(include=['object']).columns
data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Standardize the features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_encoded)

# Perform Hierarchical Clustering
# Compute linkage matrix
linkage_matrix = linkage(data_scaled, method='ward')  # "ward", "single", "complete", "average"

# Determine clusters
n_clusters = 3  # Adjust as needed
cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

# Calculate Silhouette Score
silhouette_avg = silhouette_score(data_scaled, cluster_labels)

# Output results
print(f"Number of Clusters: {n_clusters}")
print(f"Silhouette Score: {silhouette_avg:.4f}")

# Plot Dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix, truncate_mode='lastp', p=30, show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram', fontsize=14)
plt.xlabel('Cluster Size', fontsize=12)
plt.ylabel('Distance', fontsize=12)
plt.show()

# Scatter Plot (if data has 2D or is reduced to 2D using PCA)
plt.figure(figsize=(10, 6))
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=cluster_labels, cmap='viridis', s=50)
plt.title(f"Hierarchical Clustering (k={n_clusters}), Silhouette Score={silhouette_avg:.4f}", fontsize=14)
plt.xlabel('Feature 1 (Standardized)', fontsize=12)
plt.ylabel('Feature 2 (Standardized)', fontsize=12)
plt.grid(True)
plt.show()
