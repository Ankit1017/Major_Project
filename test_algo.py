import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import random

# Load the dataset
file_path = "corrected_dataset.csv"
data = pd.read_csv(file_path)
# try:
#     data = pd.read_csv(file_path, error_bad_lines=False, warn_bad_lines=True)
# except Exception as e:
#     print(f"Error while loading the dataset: {e}")

# Optional: Save the corrected dataset for review
# data.to_csv("corrected_dataset.csv", index=False)
# Data Preprocessing
# Dropping irrelevant columns (if applicable, e.g., IDs, dates)
data = data.select_dtypes(include=[np.number])  # Keep only numerical columns
data.dropna(inplace=True)  # Drop rows with missing values

# Normalize numerical data
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

# Define Gravitational Search Algorithm (GSA) for clustering
class GSAClustering:
    def __init__(self, data, n_clusters, n_agents, max_iter, G0=100, alpha=20):
        self.data = data
        self.n_clusters = n_clusters
        self.n_agents = n_agents
        self.max_iter = max_iter
        self.G0 = G0
        self.alpha = alpha
        self.agents = [self.initialize_agent() for _ in range(n_agents)]
        self.best_agent = None
        self.best_fitness = -1

    def initialize_agent(self):
        """Randomly initialize agent positions (centroids)."""
        return np.array([self.data[random.randint(0, len(self.data) - 1)] for _ in range(self.n_clusters)])

    def fitness(self, agent):
        """Calculate fitness of an agent using the Silhouette Coefficient."""
        labels = self.assign_clusters(agent)
        if len(set(labels)) < 2:  # Silhouette requires at least 2 clusters
            return -1
        print("current fitness : ",silhouette_score(self.data, labels))
        return silhouette_score(self.data, labels)

    def assign_clusters(self, centroids):
        """Assign each data point to the nearest centroid."""
        distances = np.linalg.norm(self.data[:, np.newaxis] - centroids, axis=2)
        return np.argmin(distances, axis=1)

    def update_agents(self, G):
        """Update agent positions based on gravitational forces."""
        masses = np.array([self.fitness(agent) for agent in self.agents])
        masses = (masses - masses.min()) / (masses.max() - masses.min() + 1e-9)  # Normalize masses
        total_mass = masses.sum()

        # Compute forces
        new_agents = []
        for i, agent in enumerate(self.agents):
            force = np.zeros_like(agent)
            for j, other_agent in enumerate(self.agents):
                if i != j:
                    distance = np.linalg.norm(agent - other_agent) + 1e-9
                    force += G * (masses[j] * masses[i]) / distance * (other_agent - agent)
            acceleration = force / (masses[i] + 1e-9)
            new_agents.append(agent + np.random.random() * acceleration)  # Update position
        self.agents = new_agents

    def optimize(self):
        """Run the optimization process."""
        for iteration in range(self.max_iter):
            G = self.G0 * np.exp(-self.alpha * iteration / self.max_iter)
            self.update_agents(G)
            for agent in self.agents:
                fitness = self.fitness(agent)
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_agent = agent
        return self.best_agent, self.best_fitness

# Run GSA Clustering
n_clusters = 10  # Set the desired number of clusters
n_agents =   # Number of agents
max_iter = 50  # Maximum number of iterations

gsa = GSAClustering(data_normalized, n_clusters=n_clusters, n_agents=n_agents, max_iter=max_iter)
best_centroids, best_fitness = gsa.optimize()

# Assign clusters to the data
final_labels = gsa.assign_clusters(best_centroids)

# Results
print("Best Fitness (Silhouette Coefficient):", best_fitness)
print("Cluster Centers:\n", best_centroids)

# Add cluster labels to the original data
data['Cluster'] = final_labels
