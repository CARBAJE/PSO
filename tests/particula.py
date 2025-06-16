class Particle:
    def __init__(self, id, position, speed, inertia, c1, c2):
        self.id = id
        self.position = position
        self.speed = speed
        self.inertia = inertia
        self.c1 = c1
        self.c2 = c2
        self.bestPosition = position
        self.tipo = 'Global'

    def to_list(self):
        position_list = self.position.tolist() if hasattr(self.position, 'tolist') else self.position
        speed_list = self.speed.tolist() if hasattr(self.speed, 'tolist') else self.speed
        bestPos_list  = self.bestPosition.tolist() if hasattr(self.bestPosition, 'tolist') else self.bestPosition

        return[self.id, position_list, speed_list, self.inertia, self.c1, self.c2, bestPos_list]

    def updateSpeed(self, r1, r2, bestGlobal):
        self.speed = self.inertia * self.speed + self.c1 * r1 * (self.bestPosition - self.position) + self.c2 * r2 * (bestGlobal - self.position)

    def updatePosition(self):
        self.position = self.position + self.velocity

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
    def __init__(self, id, position, speed, inertia, c1, c2, bestPosition, num_neighbors=2):
        super.__init__(id, position, speed, inertia, c1, c2, bestPosition)
        self.num_neighbors = num_neighbors
        self.neighbors = []
        self.bestNeighbor = None
        self.tipo = 'Local'

    def buildNeighborhood(self, population, N):
        if self.num_neighbors % 2 != 0:
            raise ValueError("The number of neighbors must be pair")
        middle = self.num_neighbors // 2

        lower = [(self.id - i - 1) % N + 1 for i in range(middle)]
        upper = [(self.id + i) % N + 1 for i in range(middle)]

        idx = lower + upper
        id_particles = {p.id : p for p in population}

        self.neighbors = [id_particles[i] for i in idx]

    def to_list(self):
        base_list = super.to_list()
        base_list.append([particle.id for particle in self.neighbors])

        return base_list

    def findBestNeighbor(self, f):
        bestOwnPosition = self.updateBestOwn(f)
        bestNeighbor = self.bestNeighbor

        ownBestFitness = f(bestOwnPosition)
        bestFitness = f(bestNeighbor)

        if ownBestFitness < bestFitness:
            bestFitness = ownBestFitness

        for neighbor in self.neighbors:
            fitness_neighbor = f(neighbor.bestPosition)
            if fitness_neighbor < bestFitness:
                bestFitness = fitness_neighbor
                bestNeighbor = self.position

        self.bestNeighbor = bestNeighbor

        return bestFitness

    def updateSpeed(self, r1, r2):
        self.speed = self.inertia * self.speed + self.c1 * r1 * (self.bestPosition - self.position) + self.c2 * r2 * (self.bestNeighbor - self.position)
