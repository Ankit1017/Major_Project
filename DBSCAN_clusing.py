import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('corrected_dataset.csv')  # Replace 'tr1.csv' with your dataset path

# Preprocess data: Convert categorical data to numeric if necessary
categorical_columns = data.select_dtypes(include=['object']).columns
data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_encoded)

# Tune DBSCAN parameters to get ~3 clusters
epsilon = 1.3  # Adjusted based on dataset
min_samples = 3  # Adjust as needed for cluster density

dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
cluster_labels = dbscan.fit_predict(data_scaled)

# Check number of clusters (excluding noise, which is labeled as -1)
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_noise = list(cluster_labels).count(-1)

# Calculate Silhouette Score if there are clusters
if n_clusters > 1:
    silhouette_avg = silhouette_score(data_scaled, cluster_labels)
else:
    silhouette_avg = -1  # Silhouette Score is not defined for a single cluster

# Print results
print(f"Number of Clusters: {n_clusters}")
print(f"Number of Noise Points: {n_noise}")
print(f"Silhouette Score: {silhouette_avg:.4f}" if silhouette_avg != -1 else "Silhouette Score not defined for single cluster")

# Plotting the results
plt.figure(figsize=(10, 6))
unique_labels = set(cluster_labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for label, color in zip(unique_labels, colors):
    if label == -1:  # Noise
        color = [0, 0, 0, 1]  # Black for noise

    mask = (cluster_labels == label)
    plt.scatter(data_scaled[mask, 0], data_scaled[mask, 1], c=[color], s=50, label=f"Cluster {label}" if label != -1 else "Noise")

plt.title(f"DBSCAN Clustering (eps={epsilon}, min_samples={min_samples})", fontsize=14)
plt.xlabel("Feature 1 (Standardized)", fontsize=12)
plt.ylabel("Feature 2 (Standardized)", fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
