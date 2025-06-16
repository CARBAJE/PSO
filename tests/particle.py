import numpy as np

class Particle:
    def __init__(self, id, position, speed, inertia, c1, c2):
        self.id = id
        self.position = position
        self.speed = speed
        self.inertia = inertia
        self.c1 = c1
        self.c2 = c2
        self.bestPosition = position.copy()
        self.tipo = 'Global'

    def to_list(self):
        position_list = self.position.tolist() if hasattr(self.position, 'tolist') else self.position
        speed_list = self.speed.tolist() if hasattr(self.speed, 'tolist') else self.speed
        bestPos_list  = self.bestPosition.tolist() if hasattr(self.bestPosition, 'tolist') else self.bestPosition

        return[self.id, position_list, speed_list, self.inertia, self.c1, self.c2, bestPos_list]

    def updateSpeed(self, r1, r2, bestGlobal):
        self.speed = self.inertia * self.speed + self.c1 * r1 * (self.bestPosition - self.position) + self.c2 * r2 * (bestGlobal - self.position)

    def updatePosition(self):
        self.position += self.speed

    def evaluateFitness(self, f):
        return f(self.position)

    def updateBestOwn(self, f):
        current_fitness = self.evaluateFitness(f)
        bestFitness = f(self.bestPosition)

        if current_fitness < bestFitness:
            self.bestPosition = self.position.copy()
            return True

        return False

class LocalParticle(Particle):
    def __init__(self, id, position, speed, inertia, c1, c2, num_neighbors=2):
        super().__init__(id, position, speed, inertia, c1, c2)
        self.num_neighbors = num_neighbors
        self.neighbors = []
        self.bestNeighbor = None
        self.tipo = 'Local'

    def buildNeighborhood(self, population, N):
        if self.num_neighbors % 2 != 0:
            raise ValueError("The number of neighbors must be pair")
        middle = self.num_neighbors // 2

        current_index = self.id - 1
        lower = [(current_index - i - 1) % N for i in range(middle)]
        upper = [(current_index + i + 1) % N for i in range(middle)]

        idx = lower + upper
        self.neighbors = [population[i] for i in idx]

    def to_list(self):
        base_list = super().to_list()
        base_list.append([particle.id for particle in self.neighbors])
        return base_list

    def findBestNeighbor(self, f):
        self.updateBestOwn(f)

        best_neighbor_position = self.bestPosition.copy()
        best_fitness = f(best_neighbor_position)

        for neighbor in self.neighbors:
            fitness_neighbor = f(neighbor.bestPosition)
            if fitness_neighbor < best_fitness:
                best_fitness = fitness_neighbor
                best_neighbor_position = neighbor.bestPosition.copy()

        self.bestNeighbor = best_neighbor_position

    def updateSpeed(self, r1, r2):
        self.speed = self.inertia * self.speed + self.c1 * r1 * (self.bestPosition - self.position) + self.c2 * r2 * (self.bestNeighbor - self.position)

class AdaptivePSOParticle:
    def __init__(self, id, position, speed, inertia_mean=0.5, inertia_std=0.1, c1=2.0, c2=2.0):
        self.id = id
        self.position = np.array(position, dtype=float)
        self.speed = np.array(speed, dtype=float)
        self.bestPosition = self.position.copy()
        self.inertia_mean = inertia_mean
        self.inertia_std = inertia_std
        self.c1 = c1
        self.c2 = c2
        self.success_count = 0
        self.current_inertia = self.generate_inertia()
        self.previous_fitness = float('inf')

    def generate_inertia(self):
        inertia = self.inertia_mean + self.inertia_std * np.random.randn()
        return np.clip(inertia, 1e-10, 0.9999)

    def update_inertia_mean(self, fitness_improved):
        if fitness_improved:
            new_mean = (self.inertia_mean * self.success_count + self.current_inertia) / (self.success_count + 1)
            self.inertia_mean = new_mean
            self.success_count += 1

    def updateSpeed(self, r1, r2, bestGlobal):
        cognitive = self.c1 * r1 * (self.bestPosition - self.position)
        social = self.c2 * r2 * (bestGlobal - self.position)
        self.speed = self.current_inertia * self.speed + cognitive + social

    def updatePosition(self):
        self.position += self.speed

    def evaluateFitness(self, f):
        return f(self.position)

    def updateBestOwn(self, f):
        current_fitness = self.evaluateFitness(f)
        best_fitness = f(self.bestPosition)
        fitness_improved = current_fitness < best_fitness
        if fitness_improved:
            self.bestPosition = self.position.copy()

        self.update_inertia_mean(fitness_improved)
        self.current_inertia = self.generate_inertia()
        self.previous_fitness = current_fitness

        return fitness_improved

class AdaptivePSOLocal(AdaptivePSOParticle):
    def __init__(self, id, position, speed, inertia_mean=0.5, inertia_std=0.1,
                 c1=2.0, c2=2.0, num_neighbors=2):
        super().__init__(id, position, speed, inertia_mean, inertia_std, c1, c2)
        self.num_neighbors = num_neighbors
        self.neighbors = []
        self.bestNeighbor = self.position.copy()

    def buildNeighborhood(self, population, N):
        if self.num_neighbors % 2 != 0:
            raise ValueError("El número de vecinos debe ser par")
        middle = self.num_neighbors // 2
        current_index = self.id - 1
        lower = [(current_index - i - 1) % N for i in range(middle)]
        upper = [(current_index + i + 1) % N for i in range(middle)]
        idx = lower + upper
        self.neighbors = [population[i] for i in idx]

    def findBestNeighbor(self, f):
        self.updateBestOwn(f) # Actualiza su propia mejor posición

        best_fitness = self.evaluateFitness(f)
        best_neighbor_position = self.bestPosition.copy()

        for neighbor in self.neighbors:
            neighbor_fitness = f(neighbor.bestPosition)
            if neighbor_fitness < best_fitness:
                best_fitness = neighbor_fitness
                best_neighbor_position = neighbor.bestPosition.copy()

        self.bestNeighbor = best_neighbor_position

    def updateSpeed(self, r1, r2, bestGlobal=None):
        cognitive = self.c1 * r1 * (self.bestPosition - self.position)
        social = self.c2 * r2 * (self.bestNeighbor - self.position)
        self.speed = self.current_inertia * self.speed + cognitive + social