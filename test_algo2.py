import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

class HybridGSA_SA:
    def __init__(self, data, n_agents, n_iterations, temp, cooling_rate):
        """
        Initialize the Hybrid GSA + SA.
        :param data: Input dataset.
        :param n_agents: Number of agents (solutions).
        :param n_iterations: Maximum number of iterations.
        :param temp: Initial temperature for SA.
        :param cooling_rate: Cooling rate for SA.
        """
        self.data = data
        self.n_agents = n_agents
        self.n_iterations = n_iterations
        self.temp = temp
        self.cooling_rate = cooling_rate

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

    def simulated_annealing(self, agent, current_fitness):
        """
        Perform local optimization using Simulated Annealing.
        :param agent: Current agent (centroid positions).
        :param current_fitness: Fitness of the current agent.
        :return: Optimized agent.
        """
        temp = self.temp
        best_agent = agent
        best_fitness = current_fitness
        counter=10
        while (temp > 1) & (counter>0) :
            # Generate a neighboring solution
            new_agent = agent + np.random.normal(0, 0.1, size=agent.shape)
            labels = self.assign_clusters(new_agent)

            if len(set(labels)) > 1:
                new_fitness = silhouette_score(self.data, labels)
            else:
                new_fitness = -1  # Invalid clustering

            # Accept the new solution with Metropolis criterion
            if new_fitness > best_fitness or np.exp((new_fitness - best_fitness) / temp) > np.random.rand():
                best_agent = new_agent
                best_fitness = new_fitness

            # Cool down the temperature
            temp *= self.cooling_rate
            counter-=1
        return best_agent

    def fit(self, n_clusters):
        """
        Run the Hybrid GSA + SA to find optimal clusters.
        :param n_clusters: Number of clusters.
        :return: Best centroids, best fitness, performance over iterations.
        """
        agents = self.initialize_agents(n_clusters)
        fitness = self.calculate_fitness(agents)

        best_agent = agents[np.argmax(fitness)]
        best_fitness = np.max(fitness)
        performance_over_time = []

        for iteration in range(self.n_iterations):
            # Calculate masses based on fitness
            min_fitness = np.min(fitness)
            max_fitness = np.max(fitness)
            masses = (fitness - min_fitness) / (max_fitness - min_fitness + 1e-6)

            # Update agent positions using GSA
            agents = self.update_positions(agents, masses, iteration)

            # Apply Simulated Annealing for local optimization
            for i in range(len(agents)):
                agents[i] = self.simulated_annealing(agents[i], fitness[i])

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

# Run the Hybrid GSA + SA
n_agents = 10
n_iterations = 50
n_clusters = 3
temp = 100  # Initial temperature for SA
cooling_rate = 0.95  # Cooling rate for SA

hybrid_gsa_sa = HybridGSA_SA(data_scaled, n_agents, n_iterations, temp, cooling_rate)
best_centroids, best_fitness, performance = hybrid_gsa_sa.fit(n_clusters)

# Visualize performance over iterations
plt.plot(range(1, len(performance) + 1), performance, marker="o")
plt.title("Performance Over Iterations (Hybrid GSA + SA)")
plt.xlabel("Iteration")
plt.ylabel("Fitness (Silhouette Score)")
plt.grid(True)
plt.show()
