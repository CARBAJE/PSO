import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches

class AdaptivePSOParticle:
    """
    Partícula para PSO Adaptativo según la metodología del PDF
    La inercia se actualiza usando distribución gaussiana basada en el desempeño
    """
    def __init__(self, id, position, speed, inertia_mean=0.5, inertia_std=0.1, c1=2.0, c2=2.0):
        self.id = id
        self.position = np.array(position, dtype=float)
        self.speed = np.array(speed, dtype=float)
        self.bestPosition = self.position.copy()

        # Parámetros adaptativos según PDF
        self.inertia_mean = inertia_mean  # Media de la inercia (wm en el PDF)
        self.inertia_std = inertia_std    # Desviación estándar para el muestreo gaussiano
        self.c1 = c1  # Coeficiente cognitivo
        self.c2 = c2  # Coeficiente social

        # Variables para el seguimiento adaptativo
        self.success_count = 0  # Contador de mejoras (n en el PDF)
        self.current_inertia = self.generate_inertia()  # w actual
        self.previous_fitness = float('inf')

        # Para animación - historial de posiciones
        self.position_history = [self.position.copy()]

    def generate_inertia(self):
        """
        Genera inercia usando distribución gaussiana: w = wm + σ * randn
        Según el PDF: w = μ + σ * randn donde μ=wm y σ=0.1
        """
        inertia = self.inertia_mean + self.inertia_std * np.random.randn()
        # Mantener en rango válido [0.1, 0.9]
        return np.clip(inertia, 0.1, 0.9)

    def update_inertia_mean(self, fitness_improved):
        """
        Actualiza la media de inercia según la estrategia del PDF:
        Si hay mejora: wm = (wm * n + w) / (n + 1) y n = n + 1
        """
        if fitness_improved:
            # Actualizar media usando promedio ponderado
            new_mean = (self.inertia_mean * self.success_count + self.current_inertia) / (self.success_count + 1)
            self.inertia_mean = new_mean
            self.success_count += 1

    def updateSpeed(self, r1, r2, bestReference):
        """Actualiza velocidad usando la ecuación estándar de PSO"""
        cognitive = self.c1 * r1 * (self.bestPosition - self.position)
        social = self.c2 * r2 * (bestReference - self.position)

        self.speed = self.current_inertia * self.speed + cognitive + social

    def updatePosition(self):
        """Actualiza posición de la partícula"""
        self.position = self.position + self.speed
        # Guardar para animación
        self.position_history.append(self.position.copy())

    def evaluateFitness(self, f):
        """Evalúa fitness en la posición actual"""
        if len(self.position) == 2:
            return f(self.position[0], self.position[1])
        else:
            return f(self.position)

    def updateBestOwn(self, f):
        """
        Actualiza mejor posición personal y maneja la adaptación de inercia
        Retorna True si hubo mejora en el fitness
        """
        current_fitness = self.evaluateFitness(f)

        if len(self.bestPosition) == 2:
            best_fitness = f(self.bestPosition[0], self.bestPosition[1])
        else:
            best_fitness = f(self.bestPosition)

        fitness_improved = current_fitness < best_fitness

        if fitness_improved:
            self.bestPosition = self.position.copy()

        # Actualizar media de inercia basada en el desempeño
        self.update_inertia_mean(fitness_improved)

        # Generar nueva inercia para la siguiente iteración
        self.current_inertia = self.generate_inertia()

        self.previous_fitness = current_fitness
        return fitness_improved

class AdaptivePSOLocalParticle(AdaptivePSOParticle):
    """
    Partícula para PSO Adaptativo Local según metodología del PDF
    Combina adaptación de inercia con topología local
    """
    def __init__(self, id, position, speed, inertia_mean=0.5, inertia_std=0.1,
                 c1=2.0, c2=2.0, num_neighbors=2):
        super().__init__(id, position, speed, inertia_mean, inertia_std, c1, c2)
        self.num_neighbors = num_neighbors
        self.neighbors = []
        self.bestNeighbor = self.position.copy()
        self.neighbor_ids = []  # Para visualización

    def buildNeighborhood(self, population, N):
        """Construye vecindario usando topología double-linked del PDF"""
        if self.num_neighbors % 2 != 0:
            raise ValueError("El número de vecinos debe ser par")

        middle = self.num_neighbors // 2
        lower = [(self.id - i - 1) % N for i in range(middle)]
        upper = [(self.id + i + 1) % N for i in range(middle)]

        self.neighbor_ids = lower + upper
        self.neighbors = [population[i] for i in self.neighbor_ids]

    def findBestNeighbor(self, f):
        """Encuentra el mejor vecino en el vecindario"""
        if not self.neighbors:
            self.bestNeighbor = self.bestPosition.copy()
            return self.evaluateFitness(f)

        # Incluir a sí misma en la comparación
        best_fitness = f(self.bestPosition[0], self.bestPosition[1])
        best_neighbor_position = self.bestPosition.copy()

        # Comparar con vecinos
        for neighbor in self.neighbors:
            neighbor_fitness = f(neighbor.bestPosition[0], neighbor.bestPosition[1])
            if neighbor_fitness < best_fitness:
                best_fitness = neighbor_fitness
                best_neighbor_position = neighbor.bestPosition.copy()

        self.bestNeighbor = best_neighbor_position
        return best_fitness

    def updateSpeed(self, r1, r2, bestReference=None):
        """Actualiza velocidad usando mejor vecino en lugar de mejor global"""
        cognitive = self.c1 * r1 * (self.bestPosition - self.position)
        social = self.c2 * r2 * (self.bestNeighbor - self.position)

        self.speed = self.current_inertia * self.speed + cognitive + social

class AdaptivePSOLocalAnimated:
    """
    Algoritmo PSO Adaptativo Local con capacidades de animación avanzada
    """
    def __init__(self, num_particles=20, dimensions=2, bounds=(-512, 512),
                 max_iterations=200, num_neighbors=4, inertia_mean=0.5):
        self.num_particles = num_particles
        self.dimensions = dimensions
        self.bounds = bounds
        self.max_iterations = max_iterations
        self.num_neighbors = num_neighbors
        self.inertia_mean = inertia_mean

        # Variables de seguimiento
        self.best_global_position = None
        self.best_global_fitness = float('inf')
        self.fitness_history = []
        self.inertia_history = []
        self.best_position_history = []

        # Para animación
        self.iteration_data = []
        self.neighborhood_connections = []

        # Inicializar población (debe ser al final)
        self.population = self._initialize_population()

        # Construir conexiones de vecindario después de tener la población
        self._build_neighborhood_connections()

    def _initialize_population(self):
        """Inicializa población de partículas locales"""
        population = []

        for i in range(self.num_particles):
            # Posición aleatoria en el dominio
            position = np.random.uniform(self.bounds[0], self.bounds[1], self.dimensions)

            # Velocidad inicial aleatoria
            d = self.bounds[1] - self.bounds[0]
            speed = np.random.uniform(-d/10, d/10, self.dimensions)

            particle = AdaptivePSOLocalParticle(i, position, speed, self.inertia_mean,
                                              num_neighbors=self.num_neighbors)
            population.append(particle)

        # Construir vecindarios
        for particle in population:
            particle.buildNeighborhood(population, self.num_particles)

        return population

    def _build_neighborhood_connections(self):
        """Construye lista de conexiones de vecindario para visualización"""
        self.neighborhood_connections = []
        for particle in self.population:
            for neighbor_id in particle.neighbor_ids:
                # Evitar duplicados (conexión bidireccional)
                connection = tuple(sorted([particle.id, neighbor_id]))
                if connection not in self.neighborhood_connections:
                    self.neighborhood_connections.append(connection)

    def _enforce_bounds(self, particle):
        """Corrige violaciones de dominio según el PDF"""
        for j in range(self.dimensions):
            if particle.position[j] < self.bounds[0] or particle.position[j] > self.bounds[1]:
                # Generar nueva posición dentro del rango
                particle.position[j] = np.random.uniform(self.bounds[0], self.bounds[1])

                # Ajustar velocidad
                d = self.bounds[1] - self.bounds[0]
                particle.speed[j] = np.random.uniform(-d/10, d/10)

    def _update_global_best(self, objective_function):
        """Actualiza mejor posición global (para seguimiento)"""
        for particle in self.population:
            fitness = particle.evaluateFitness(objective_function)
            if fitness < self.best_global_fitness:
                self.best_global_fitness = fitness
                self.best_global_position = particle.position.copy()

    def optimize_with_animation_data(self, objective_function, verbose=True):
        """
        Ejecuta optimización PSO adaptativo local guardando datos para animación
        """
        for iteration in range(self.max_iterations):
            # Actualizar mejores posiciones personales
            for particle in self.population:
                particle.updateBestOwn(objective_function)

            # Encontrar mejores vecinos para cada partícula
            for particle in self.population:
                particle.findBestNeighbor(objective_function)

            # Actualizar mejor posición global (solo para seguimiento)
            self._update_global_best(objective_function)

            # Guardar datos de la iteración actual para animación
            current_positions = np.array([p.position for p in self.population])
            current_best_positions = np.array([p.bestPosition for p in self.population])
            current_best_neighbors = np.array([p.bestNeighbor for p in self.population])
            current_fitnesses = np.array([p.evaluateFitness(objective_function) for p in self.population])
            current_inertias = np.array([p.current_inertia for p in self.population])

            self.iteration_data.append({
                'iteration': iteration,
                'positions': current_positions.copy(),
                'best_positions': current_best_positions.copy(),
                'best_neighbors': current_best_neighbors.copy(),
                'fitnesses': current_fitnesses.copy(),
                'inertias': current_inertias.copy(),
                'global_best': self.best_global_position.copy() if self.best_global_position is not None else None,
                'global_best_fitness': self.best_global_fitness
            })

            # Actualizar velocidades y posiciones
            for particle in self.population:
                r1 = np.random.random(self.dimensions)
                r2 = np.random.random(self.dimensions)

                particle.updateSpeed(r1, r2)
                particle.updatePosition()
                self._enforce_bounds(particle)

            # Registrar progreso
            self.fitness_history.append(self.best_global_fitness)
            avg_inertia = np.mean([p.inertia_mean for p in self.population])
            self.inertia_history.append(avg_inertia)

            if self.best_global_position is not None:
                self.best_position_history.append(self.best_global_position.copy())

            if verbose and iteration % 50 == 0:
                print(f"Iteración {iteration}: Mejor fitness = {self.best_global_fitness:.6f}, "
                      f"Inercia promedio = {avg_inertia:.4f}")

        return self.best_global_position, self.best_global_fitness

    def create_animation_2d_local(self, objective_function, interval=100, trail_length=10,
                                show_neighborhoods=True):
        """
        Crea animación 2D especializada para PSO Local con visualización de vecindarios
        """
        if not self.iteration_data:
            raise ValueError("Primero debe ejecutar optimize_with_animation_data()")

        # Crear malla para el mapa de contorno
        x = np.linspace(self.bounds[0], self.bounds[1], 100)
        y = np.linspace(self.bounds[0], self.bounds[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = objective_function(X, Y)

        # Configurar figura con 3 subplots
        fig = plt.figure(figsize=(20, 6))

        # Subplot 1: Animación de partículas con vecindarios
        ax1 = plt.subplot(131)
        im = ax1.contour(X, Y, Z, levels=20, colors='gray', alpha=0.4)
        ax1.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)

        # Inicializar elementos de la animación
        particles_scat = ax1.scatter([], [], c='red', s=40, alpha=0.8, label='Partículas', zorder=5)
        best_positions_scat = ax1.scatter([], [], c='blue', s=25, alpha=0.6,
                                        label='Mejores personales', zorder=4)
        best_neighbors_scat = ax1.scatter([], [], c='orange', s=30, marker='s', alpha=0.7,
                                        label='Mejores vecinos', zorder=4)
        global_best_scat = ax1.scatter([], [], c='gold', s=120, marker='*',
                                     edgecolors='black', linewidth=2,
                                     label='Mejor global', zorder=6)

        # Líneas de vecindario
        neighborhood_lines = []
        if show_neighborhoods:
            for _ in self.neighborhood_connections:
                line, = ax1.plot([], [], 'gray', alpha=0.3, linewidth=1, zorder=1)
                neighborhood_lines.append(line)

        # Líneas de trayectoria
        trail_lines = []
        for i in range(self.num_particles):
            line, = ax1.plot([], [], 'r-', alpha=0.3, linewidth=0.5, zorder=2)
            trail_lines.append(line)

        ax1.set_xlim(self.bounds[0], self.bounds[1])
        ax1.set_ylim(self.bounds[0], self.bounds[1])
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title('PSO Adaptativo Local - Evolución 2D')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Subplot 2: Convergencia en tiempo real
        ax2 = plt.subplot(132)
        fitness_line, = ax2.plot([], [], 'b-', linewidth=2, label='Mejor fitness')
        ax2_twin = ax2.twinx()
        inertia_line, = ax2_twin.plot([], [], 'r--', linewidth=2, label='Inercia promedio')

        ax2.set_xlabel('Iteración')
        ax2.set_ylabel('Fitness', color='b')
        ax2_twin.set_ylabel('Inercia promedio', color='r')
        ax2.set_title('Convergencia')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')
        ax2_twin.legend(loc='upper left')

        # Subplot 3: Distribución de inercias
        ax3 = plt.subplot(133)

        # Texto de información
        info_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes,
                           verticalalignment='top', fontsize=9,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        def animate(frame):
            if frame >= len(self.iteration_data):
                return particles_scat, best_positions_scat, global_best_scat

            data = self.iteration_data[frame]

            # Actualizar posiciones de partículas
            particles_scat.set_offsets(data['positions'])
            best_positions_scat.set_offsets(data['best_positions'])
            best_neighbors_scat.set_offsets(data['best_neighbors'])

            # Actualizar mejor global
            if data['global_best'] is not None:
                global_best_scat.set_offsets([data['global_best']])

            # Actualizar líneas de vecindario
            if show_neighborhoods and neighborhood_lines:
                for i, (p1_id, p2_id) in enumerate(self.neighborhood_connections):
                    if i < len(neighborhood_lines):
                        p1_pos = data['positions'][p1_id]
                        p2_pos = data['positions'][p2_id]
                        neighborhood_lines[i].set_data([p1_pos[0], p2_pos[0]],
                                                     [p1_pos[1], p2_pos[1]])

            # Actualizar trayectorias
            for i, line in enumerate(trail_lines):
                if i < len(self.population):
                    history = self.population[i].position_history
                    if len(history) > 1:
                        start_idx = max(0, len(history) - trail_length)
                        trail_positions = history[start_idx:frame+1] if frame < len(history) else history[start_idx:]
                        if len(trail_positions) > 1:
                            trail_array = np.array(trail_positions)
                            line.set_data(trail_array[:, 0], trail_array[:, 1])

            # Actualizar gráficos de convergencia
            iterations = list(range(frame + 1))
            fitness_values = self.fitness_history[:frame + 1]
            inertia_values = self.inertia_history[:frame + 1]

            if fitness_values:
                fitness_line.set_data(iterations, fitness_values)
                ax2.set_xlim(0, max(10, frame))
                if len(fitness_values) > 1:
                    y_min, y_max = min(fitness_values), max(fitness_values)
                    margin = abs(y_max - y_min) * 0.1 + 1e-6
                    ax2.set_ylim(y_min - margin, y_max + margin)

            if inertia_values:
                inertia_line.set_data(iterations, inertia_values)
                ax2_twin.set_xlim(0, max(10, frame))
                if len(inertia_values) > 1:
                    y_min, y_max = min(inertia_values), max(inertia_values)
                    margin = abs(y_max - y_min) * 0.1 + 1e-6
                    ax2_twin.set_ylim(y_min - margin, y_max + margin)

            # Actualizar histograma de inercias
            ax3.clear()
            if 'inertias' in data:
                ax3.hist(data['inertias'], bins=10, alpha=0.7, color='green', edgecolor='black')
                ax3.set_xlabel('Inercia')
                ax3.set_ylabel('Frecuencia')
                ax3.set_title('Distribución de Inercias')
                ax3.grid(True, alpha=0.3)

            # Actualizar información
            diversity = np.std(data['positions'], axis=0).mean()
            info_text.set_text(f"Iteración: {frame}\n"
                             f"Mejor fitness: {data['global_best_fitness']:.4f}\n"
                             f"Partículas: {len(data['positions'])}\n"
                             f"Vecinos por partícula: {self.num_neighbors}\n"
                             f"Diversidad: {diversity:.4f}")

            return (particles_scat, best_positions_scat, best_neighbors_scat,
                   global_best_scat, info_text, fitness_line, inertia_line) + tuple(trail_lines) + tuple(neighborhood_lines)

        anim = FuncAnimation(fig, animate, frames=len(self.iteration_data),
                           interval=interval, blit=False, repeat=True)

        plt.tight_layout()
        return fig, anim

# Funciones de prueba
def eggholder(x, y):
    """Función Eggholder para pruebas"""
    term1 = -(y + 47) * np.sin(np.sqrt(np.abs(x/2 + (y + 47))))
    term2 = -x * np.sin(np.sqrt(np.abs(x - (y + 47))))
    return term1 + term2

def rastrigin(x, y):
    """Función Rastrigin para pruebas"""
    A = 10
    return A * 2 + (x**2 - A * np.cos(2 * np.pi * x)) + (y**2 - A * np.cos(2 * np.pi * y))

def ackley(x, y):
    """Función Ackley para pruebas"""
    return (-20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) -
            np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) +
            np.e + 20)

def ejemplo_pso_local_animado():
    """Ejemplo de uso del PSO Adaptativo Local con animación"""
    print("=== PSO Adaptativo LOCAL con Animación ===\n")

    # Función a optimizar
    func_name = "Eggholder"
    objective_function = eggholder
    bounds = (-512, 512)

    print(f"Optimizando función: {func_name}")
    print(f"Dominio: [{bounds[0]}, {bounds[1]}]")

    # Crear optimizador LOCAL
    pso_local = AdaptivePSOLocalAnimated(
        num_particles=30,
        dimensions=2,
        bounds=bounds,
        max_iterations=200,
        num_neighbors=4,  # Vecindario de 4 partículas
        inertia_mean=0.6
    )

    # Ejecutar optimización
    print("Ejecutando optimización LOCAL...")
    best_pos, best_fit = pso_local.optimize_with_animation_data(objective_function, verbose=True)

    print(f"\nResultados finales:")
    print(f"Mejor posición: {best_pos}")
    print(f"Mejor fitness: {best_fit:.6f}")
    print(f"Número de vecinos por partícula: {pso_local.num_neighbors}")
    print(f"Total de conexiones de vecindario: {len(pso_local.neighborhood_connections)}")

    # Crear animación especializada para PSO Local
    print("\nCreando animación PSO Local...")
    fig_local, anim_local = pso_local.create_animation_2d_local(
        objective_function,
        interval=80,
        trail_length=12,
        show_neighborhoods=True
    )

    # Mostrar animación
    plt.show()

    return pso_local, anim_local

if __name__ == "__main__":
    # Ejecutar ejemplo
    pso_optimizer, animation_local = ejemplo_pso_local_animado()

    # Para guardar la animación (opcional)
    # animation_local.save('pso_local_animation.gif', writer='pillow', fps=8)