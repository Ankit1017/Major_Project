import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd 
class GSAWithEliteAgents:
    def __init__(self, data, n_agents, n_iterations, elite_fraction):
        """
        Initialize the GSA with Elite Agents.
        :param data: Input dataset.
        :param n_agents: Number of agents (solutions).
        :param n_iterations: Maximum number of iterations.
        :param elite_fraction: Fraction of agents considered as elites.
        """
        self.data = data
        self.n_agents = n_agents
        self.n_iterations = n_iterations
        self.n_elites = max(1, int(n_agents * elite_fraction))  # At least 1 elite agent

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

    def update_positions(self, agents, fitness, iteration):
        """
        Update the positions of agents based on gravitational forces from elites.
        :param agents: Current agent positions.
        :param fitness: Current fitness values of agents.
        :param iteration: Current iteration number.
        :return: Updated agent positions.
        """
        G = 1 / (1 + iteration)  # Gravitational constant decreases over time
        sorted_indices = np.argsort(fitness)[::-1]
        elites = agents[sorted_indices[:self.n_elites]]  # Select elite agents
        best_fitness = fitness[sorted_indices[0]]

        new_agents = np.copy(agents)
        for i, agent in enumerate(agents):
            force = np.zeros(agent.shape)
            for elite in elites:
                r = np.linalg.norm(agent - elite)
                if r > 0:
                    force += G * (best_fitness - fitness[i]) * (elite - agent) / r
            new_agents[i] += np.random.rand() * force  # Update position
        return new_agents

    def fit(self, n_clusters):
        """
        Run the GSA with Elite Agents to find optimal clusters.
        :param n_clusters: Number of clusters.
        :return: Best centroids, best fitness, performance over iterations.
        """
        agents = self.initialize_agents(n_clusters)
        fitness = self.calculate_fitness(agents)

        best_agent = agents[np.argmax(fitness)]
        best_fitness = np.max(fitness)
        performance_over_time = []

        for iteration in range(self.n_iterations):
            # Update agent positions
            agents = self.update_positions(agents, fitness, iteration)

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

# Run the GSA with Elite Agents
n_agents = 10
n_iterations = 100
elite_fraction = 0.2  # 20% of agents are elites
n_clusters = 3

gsa_elite = GSAWithEliteAgents(data_scaled, n_agents, n_iterations, elite_fraction)
best_centroids, best_fitness, performance = gsa_elite.fit(n_clusters)

# Visualize performance over iterations
plt.plot(range(1, len(performance) + 1), performance, marker="o")
plt.title("Performance Over Iterations with Elite Agents (GSA)")
plt.xlabel("Iteration")
plt.ylabel("Fitness (Silhouette Score)")
plt.grid(True)
plt.show()
