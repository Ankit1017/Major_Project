import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class STING:
    def __init__(self, bins, threshold):
        """
        Initialize the STING clustering algorithm.
        :param bins: Number of bins (grid cells) for each dimension.
        :param threshold: Threshold to filter grid cells as part of a cluster.
        """
        self.bins = bins
        self.threshold = threshold

    def fit(self, data):
        """
        Fit the STING model to the dataset.
        :param data: Input dataset (numpy array).
        :return: List of cluster labels.
        """
        self.grid_size = [np.linspace(np.min(data[:, i]), np.max(data[:, i]), self.bins + 1) for i in range(data.shape[1])]
        self.grid_cells = np.zeros((self.bins,) * data.shape[1])

        # Populate grid cells with data counts
        for point in data:
            grid_indices = tuple(np.digitize(point[i], self.grid_size[i]) - 1 for i in range(data.shape[1]))
            self.grid_cells[grid_indices] += 1

        # Filter cells based on threshold
        clusters = np.zeros(len(data), dtype=int)
        current_cluster = 1
        for i, point in enumerate(data):
            grid_indices = tuple(np.digitize(point[j], self.grid_size[j]) - 1 for j in range(data.shape[1]))
            if self.grid_cells[grid_indices] >= self.threshold:
                clusters[i] = current_cluster
            else:
                clusters[i] = -1  # Mark as noise

        return clusters

# Load dataset
data = pd.read_csv('corrected_dataset.csv')  # Replace with your dataset path

# Preprocess data: Convert categorical data to numeric and standardize
categorical_columns = data.select_dtypes(include=['object']).columns
data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Apply PCA for dimensionality reduction to 2D
pca = PCA(n_components=2)
data_reduced = pca.fit_transform(data_encoded)

# Apply STING
bins = 10         # Number of bins
threshold = 5     # Threshold to filter grid cells as clusters

sting = STING(bins=bins, threshold=threshold)
cluster_labels = sting.fit(data_reduced)

# Visualize the results
unique_labels = set(cluster_labels)
plt.figure(figsize=(10, 6))
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for label, color in zip(unique_labels, colors):
    if label == -1:  # Noise
        color = [0, 0, 0, 1]  # Black for noise

    mask = (cluster_labels == label)
    plt.scatter(data_reduced[mask, 0], data_reduced[mask, 1], c=[color], s=50, label=f"Cluster {label}" if label != -1 else "Noise")

plt.title(f"STING Clustering (bins={bins}, threshold={threshold})", fontsize=14)
plt.xlabel("Principal Component 1", fontsize=12)
plt.ylabel("Principal Component 2", fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
