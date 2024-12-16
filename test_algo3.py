import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
class HybridGSA_SA_AgeDiversity:
    def __init__(self, data, n_agents, n_iterations, alpha, age_threshold):
        """
        Initialize the Hybrid GSA + SA algorithm with Age Diversity Monitoring.
        :param data: Input dataset.
        :param n_agents: Number of agents (solutions).
        :param n_iterations: Maximum number of iterations.
        :param alpha: Simulated annealing cooling rate.
        :param age_threshold: Threshold for stagnation age.
        """
        self.data = data
        self.n_agents = n_agents
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.age_threshold = age_threshold

    def initialize_agents(self, n_clusters):
        agents = np.random.rand(self.n_agents, n_clusters, self.data.shape[1])
        ages = np.zeros(self.n_agents)  # Initialize ages for all agents
        return agents, ages

    def calculate_fitness(self, agents):
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
        distances = np.linalg.norm(self.data[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        return labels

    def update_positions(self, agents, fitness, iteration):
        G = 1 / (1 + iteration)  # Gravitational constant decreases over time
        sorted_indices = np.argsort(fitness)[::-1]
        best_agents = agents[sorted_indices]
        best_fitness = fitness[sorted_indices]

        new_agents = np.copy(agents)
        for i, agent in enumerate(agents):
            force = np.zeros(agent.shape)
            for j, best_agent in enumerate(best_agents):
                if i != j:  # Avoid self-interaction
                    r = np.linalg.norm(agent - best_agent)
                    if r > 0:
                        force += G * (best_fitness[j] - best_fitness[i]) * (best_agent - agent) / r
            new_agents[i] += np.random.rand() * force  # Update position

        return new_agents

    def simulated_annealing(self, agent, best_fitness, temperature):
        new_agent = agent + np.random.normal(0, 0.1, agent.shape)
        labels = self.assign_clusters(new_agent)
        if len(set(labels)) > 1:
            new_fitness = silhouette_score(self.data, labels)
        else:
            new_fitness = -1

        if new_fitness > best_fitness or np.random.rand() < np.exp((new_fitness - best_fitness) / temperature):
            return new_agent, new_fitness
        return agent, best_fitness

    def reinitialize_agent(self, agent):
        return np.random.rand(*agent.shape)

    def fit(self, n_clusters):
        agents, ages = self.initialize_agents(n_clusters)
        fitness = self.calculate_fitness(agents)

        best_agent = agents[np.argmax(fitness)]
        best_fitness = np.max(fitness)

        temperature = 1.0  # Initial temperature for simulated annealing
        performance_over_time = []

        for iteration in range(self.n_iterations):
            # Update agent positions using GSA
            agents = self.update_positions(agents, fitness, iteration)

            # Apply simulated annealing to improve agents
            for i in range(self.n_agents):
                agents[i], fitness[i] = self.simulated_annealing(agents[i], fitness[i], temperature)

                # Monitor stagnation and reinitialize if necessary
                if fitness[i] <= fitness[i]:
                    ages[i] += 1
                else:
                    ages[i] = 0  # Reset age if fitness improves

                if ages[i] > self.age_threshold:
                    agents[i] = self.reinitialize_agent(agents[i])
                    ages[i] = 0  # Reset age after reinitialization

            # Update global best
            current_best_fitness = np.max(fitness)
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_agent = agents[np.argmax(fitness)]

            # Cooling schedule for simulated annealing
            temperature *= self.alpha

            # Track performance
            performance_over_time.append(best_fitness)
            print(f"Iteration {iteration + 1}/{self.n_iterations}, Best Fitness: {best_fitness}")

        return best_agent, best_fitness, performance_over_time

# Load dataset and preprocess
data = pd.read_csv("corrected_dataset.csv")  # Replace with your dataset path

data = data.select_dtypes(include=[np.number])  # Keep only numerical columns
data.dropna(inplace=True)  # Drop rows with missing values

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Run the Hybrid GSA + SA with Age Diversity Monitoring
n_agents = 10
n_iterations = 100
alpha = 0.95
age_threshold = 10
n_clusters = 3

hybrid_gsa_sa = HybridGSA_SA_AgeDiversity(data_scaled, n_agents, n_iterations, alpha, age_threshold)
best_centroids, best_fitness, performance = hybrid_gsa_sa.fit(n_clusters)

# Visualize performance over iterations
import matplotlib.pyplot as plt

plt.plot(range(1, len(performance) + 1), performance, marker="o")
plt.title("Performance Over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Fitness (Silhouette Score)")
plt.grid(True)
plt.show()
