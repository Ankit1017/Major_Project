import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
class GSAWithDistanceWeightedMasses:
    def __init__(self, data, n_agents, n_iterations):
        """
        Initialize the GSA with Distance-Weighted Masses.
        :param data: Input dataset.
        :param n_agents: Number of agents (solutions).
        :param n_iterations: Maximum number of iterations.
        """
        self.data = data
        self.n_agents = n_agents
        self.n_iterations = n_iterations

    def initialize_agents(self, n_clusters):
        """
        Initialize agent positions randomly.
        :param n_clusters: Number of clusters.
        :return: Initialized agent positions.
        """
        return np.random.rand(self.n_agents, n_clusters, self.data.shape[1])

    def calculate_fitness(self, agents):
        """
        Calculate the fitness (Silhouette Score) for each agent.
        :param agents: Agent positions.
        :return: Array of fitness scores.
        """
        fitness = []
        for agent in agents:
            labels = self.assign_clusters(agent)
            if len(set(labels)) > 1:  # Avoid silhouette calculation if only one cluster
                score = silhouette_score(self.data, labels)
            else:
                score = -1  # Invalid clustering
            fitness.append(score)
        return np.array(fitness)

    def assign_clusters(self, centroids):
        """
        Assign data points to the nearest centroid.
        :param centroids: Centroid positions.
        :return: Cluster labels.
        """
        distances = np.linalg.norm(self.data[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        return labels

    def calculate_distance_weighted_masses(self, agents, fitness):
        """
        Calculate masses weighted by fitness and distance.
        :param agents: Agent positions.
        :param fitness: Fitness values of agents.
        :return: Distance-weighted masses for each agent.
        """
        min_fitness = np.min(fitness)
        max_fitness = np.max(fitness)
        masses = (fitness - min_fitness) / (max_fitness - min_fitness + 1e-6)

        # Calculate pairwise distances between agents (centroids flattened for each agent)
        agents_flat = agents.reshape(self.n_agents, -1)  # Flatten centroids
        distances = np.linalg.norm(agents_flat[:, np.newaxis] - agents_flat, axis=2)
        
        # Avoid division by zero (self-influence)
        np.fill_diagonal(distances, np.inf)
        
        # Calculate distance-weighted masses
        weighted_masses = masses / (distances + 1e-6)  # Add small value to avoid division by zero
        return np.sum(weighted_masses, axis=1)


    def update_positions(self, agents, masses, iteration):
        """
        Update the positions of agents based on gravitational forces.
        :param agents: Current agent positions.
        :param masses: Masses of agents.
        :param iteration: Current iteration number.
        :return: Updated agent positions.
        """
        G = 1 / (1 + iteration)  # Gravitational constant decreases over time

        new_agents = np.copy(agents)
        for i, agent in enumerate(agents):
            force = np.zeros(agent.shape)
            for j, other_agent in enumerate(agents):
                if i != j:
                    r = np.linalg.norm(agent - other_agent)
                    if r > 0:
                        force += G * masses[j] * (other_agent - agent) / r
            new_agents[i] += np.random.rand() * force  # Update position
        return new_agents

    def fit(self, n_clusters):
        """
        Run the GSA with Distance-Weighted Masses to find optimal clusters.
        :param n_clusters: Number of clusters.
        :return: Best centroids, best fitness, performance over iterations.
        """
        agents = self.initialize_agents(n_clusters)
        fitness = self.calculate_fitness(agents)

        best_agent = agents[np.argmax(fitness)]
        best_fitness = np.max(fitness)
        performance_over_time = []

        for iteration in range(self.n_iterations):
            # Calculate distance-weighted masses
            masses = self.calculate_distance_weighted_masses(agents, fitness)

            # Update agent positions
            agents = self.update_positions(agents, masses, iteration)

            # Recalculate fitness
            fitness = self.calculate_fitness(agents)

            # Update global best
            current_best_fitness = np.max(fitness)
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_agent = agents[np.argmax(fitness)]

            # Track performance
            performance_over_time.append(best_fitness)
            print(f"Iteration {iteration + 1}/{self.n_iterations}, Best Fitness: {best_fitness}")

        return best_agent, best_fitness, performance_over_time


# Load dataset and preprocess
data = pd.read_csv("corrected_dataset.csv")
data = data.select_dtypes(include=[np.number])  # Keep only numerical columns
data.dropna(inplace=True)  # Drop rows with missing values

# Replace with your dataset path
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)


# Run the GSA with Distance-Weighted Masses
n_agents = 10
n_iterations = 100
n_clusters = 3

gsa_distance_weighted = GSAWithDistanceWeightedMasses(data_scaled, n_agents, n_iterations)
best_centroids, best_fitness, performance = gsa_distance_weighted.fit(n_clusters)

# Visualize performance over iterations
plt.plot(range(1, len(performance) + 1), performance, marker="o")
plt.title("Performance Over Iterations with Distance-Weighted Masses (GSA)")
plt.xlabel("Iteration")
plt.ylabel("Fitness (Silhouette Score)")
plt.grid(True)
plt.show()






