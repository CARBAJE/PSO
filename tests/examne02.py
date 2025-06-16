import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import time
import pandas as pd
from scipy import stats

# --- Clases originales ---
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
        self.inertia_mean = inertia_mean
        self.inertia_std = inertia_std
        self.c1 = c1
        self.c2 = c2
        self.success_count = 0
        self.current_inertia = self.generate_inertia()
        self.previous_fitness = float('inf')
        self.position_history = [self.position.copy()]

    def generate_inertia(self):
        inertia = self.inertia_mean + self.inertia_std * np.random.randn()
        return np.clip(inertia, 0.1, 0.9)

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
        self.position = self.position + self.speed
        self.position_history.append(self.position.copy())

    def evaluateFitness(self, f):
        if len(self.position) == 2:
            return f(self.position[0], self.position[1])
        else:
            return f(self.position)

    def updateBestOwn(self, f):
        current_fitness = self.evaluateFitness(f)
        if len(self.bestPosition) == 2:
            best_fitness = f(self.bestPosition[0], self.bestPosition[1])
        else:
            best_fitness = f(self.bestPosition)
        fitness_improved = current_fitness < best_fitness
        if fitness_improved:
            self.bestPosition = self.position.copy()
        self.update_inertia_mean(fitness_improved)
        self.current_inertia = self.generate_inertia()
        self.previous_fitness = current_fitness
        return fitness_improved

class AdaptivePSOLocal(AdaptivePSOParticle):
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

    def buildNeighborhood(self, population, N):
        if self.num_neighbors % 2 != 0:
            raise ValueError("El número de vecinos debe ser par")
        middle = self.num_neighbors // 2
        lower = [(self.id - i - 1) % N for i in range(middle)]
        upper = [(self.id + i + 1) % N for i in range(middle)]
        idx = lower + upper
        self.neighbors = [population[i] for i in idx]

    def findBestNeighbor(self, f):
        if not self.neighbors:
            self.bestNeighbor = self.bestPosition.copy()
            return self.evaluateFitness(f)

        best_fitness = f(self.bestPosition[0], self.bestPosition[1])
        best_neighbor_position = self.bestPosition.copy()

        for neighbor in self.neighbors:
            neighbor_fitness = f(neighbor.bestPosition[0], neighbor.bestPosition[1])

            if neighbor_fitness < best_fitness:
                best_fitness = neighbor_fitness
                best_neighbor_position = neighbor.bestPosition.copy()

        self.bestNeighbor = best_neighbor_position
        return best_fitness

    def updateSpeed(self, r1, r2, bestGlobal=None):
        cognitive = self.c1 * r1 * (self.bestPosition - self.position)
        social = self.c2 * r2 * (self.bestNeighbor - self.position)
        self.speed = self.current_inertia * self.speed + cognitive + social

# --- Función Rastrigin ---
def rastrigin(x, y):
    """
    Función Rastrigin en 2D
    Mínimo global en (0, 0) con valor 0
    """
    A = 10
    return A * 2 + (x**2 - A * np.cos(2 * np.pi * x)) + (y**2 - A * np.cos(2 * np.pi * y))

# --- Algoritmos PSO ---
class PSOAlgorithm:
    def __init__(self, particle_class, num_particles=30, dimensions=2, bounds=(-5.12, 5.12)):
        self.particle_class = particle_class
        self.num_particles = num_particles
        self.dimensions = dimensions
        self.bounds = bounds
        self.particles = []
        self.bestGlobal = None
        self.bestGlobalFitness = float('inf')
        self.fitness_history = []

    def initialize_particles(self):
        self.particles = []
        for i in range(self.num_particles):
            # Posición aleatoria dentro de los límites
            position = np.random.uniform(self.bounds[0], self.bounds[1], self.dimensions)
            # Velocidad inicial aleatoria
            speed = np.random.uniform(-1, 1, self.dimensions)

            if self.particle_class == AdaptivePSOLocal:
                particle = self.particle_class(i, position, speed, num_neighbors=4)
            else:
                particle = self.particle_class(i, position, speed)

            self.particles.append(particle)

        # Construir vecindarios para PSO Local
        if self.particle_class == AdaptivePSOLocal:
            for particle in self.particles:
                particle.buildNeighborhood(self.particles, self.num_particles)

    def update_global_best(self):
        for particle in self.particles:
            fitness = particle.evaluateFitness(rastrigin)
            if fitness < self.bestGlobalFitness:
                self.bestGlobalFitness = fitness
                self.bestGlobal = particle.position.copy()

    def run(self, max_iterations=500):
        self.initialize_particles()
        self.update_global_best()

        for iteration in range(max_iterations):
            for particle in self.particles:
                r1 = np.random.random(self.dimensions)
                r2 = np.random.random(self.dimensions)

                if self.particle_class == AdaptivePSOLocal:
                    particle.findBestNeighbor(rastrigin)
                    particle.updateSpeed(r1, r2)
                else:
                    particle.updateSpeed(r1, r2, self.bestGlobal)

                particle.updatePosition()

                # Aplicar límites
                particle.position = np.clip(particle.position, self.bounds[0], self.bounds[1])

                particle.updateBestOwn(rastrigin)

            self.update_global_best()
            self.fitness_history.append(self.bestGlobalFitness)

        return self.bestGlobal, self.bestGlobalFitness

# --- Entorno de Pruebas ---
class PSOTestEnvironment:
    def __init__(self, num_runs=10):
        self.num_runs = num_runs
        self.results = {
            'Adaptive PSO': {'best_solutions': [], 'best_fitness': [], 'convergence': [], 'times': []},
            'Adaptive PSO Local': {'best_solutions': [], 'best_fitness': [], 'convergence': [], 'times': []}
        }

    def run_tests(self):
        print("Iniciando pruebas comparativas PSO - Función Rastrigin")
        print("=" * 60)

        algorithms = [
            ('Adaptive PSO', AdaptivePSOParticle),
            ('Adaptive PSO Local', AdaptivePSOLocal)
        ]

        for alg_name, alg_class in algorithms:
            print(f"\n🔄 Ejecutando {alg_name}...")

            for run in range(self.num_runs):
                print(f"  Ejecución {run + 1}/{self.num_runs}", end=" - ")

                start_time = time.time()
                pso = PSOAlgorithm(alg_class)
                best_solution, best_fitness = pso.run()
                end_time = time.time()

                self.results[alg_name]['best_solutions'].append(best_solution)
                self.results[alg_name]['best_fitness'].append(best_fitness)
                self.results[alg_name]['convergence'].append(pso.fitness_history)
                self.results[alg_name]['times'].append(end_time - start_time)

                print(f"Fitness: {best_fitness:.6f}, Tiempo: {end_time - start_time:.2f}s")

    def calculate_statistics(self):
        stats = {}

        for alg_name in self.results.keys():
            fitness_values = self.results[alg_name]['best_fitness']
            times = self.results[alg_name]['times']

            stats[alg_name] = {
                'mean_fitness': np.mean(fitness_values),
                'std_fitness': np.std(fitness_values),
                'min_fitness': np.min(fitness_values),
                'max_fitness': np.max(fitness_values),
                'median_fitness': np.median(fitness_values),
                'mean_time': np.mean(times),
                'std_time': np.std(times),
                'success_rate': np.sum(np.array(fitness_values) < 1e-6) / len(fitness_values) * 100
            }

        return stats

    def print_statistics(self):
        stats = self.calculate_statistics()

        print("\n" + "=" * 80)
        print("ESTADÍSTICAS COMPARATIVAS")
        print("=" * 80)

        for alg_name, alg_stats in stats.items():
            print(f"\n🔸 {alg_name}:")
            print(f"   Fitness promedio:    {alg_stats['mean_fitness']:.6f} ± {alg_stats['std_fitness']:.6f}")
            print(f"   Mejor fitness:       {alg_stats['min_fitness']:.6f}")
            print(f"   Peor fitness:        {alg_stats['max_fitness']:.6f}")
            print(f"   Mediana fitness:     {alg_stats['median_fitness']:.6f}")
            print(f"   Tiempo promedio:     {alg_stats['mean_time']:.2f} ± {alg_stats['std_time']:.2f} segundos")
            print(f"   Tasa de éxito:       {alg_stats['success_rate']:.1f}% (fitness < 1e-6)")

    def statistical_test(self):
        """Prueba t de Student para comparar los resultados"""
        fitness1 = self.results['Adaptive PSO']['best_fitness']
        fitness2 = self.results['Adaptive PSO Local']['best_fitness']

        t_stat, p_value = stats.ttest_ind(fitness1, fitness2)

        print(f"\nPRUEBA ESTADÍSTICA (t-test)")
        print(f"   Estadístico t: {t_stat:.4f}")
        print(f"   Valor p:       {p_value:.6f}")

        if p_value < 0.05:
            better_alg = "Adaptive PSO" if np.mean(fitness1) < np.mean(fitness2) else "Adaptive PSO Local"
            print(f"   Conclusión:    {better_alg} es significativamente mejor (p < 0.05)")
        else:
            print(f"   Conclusión:    No hay diferencia significativa entre algoritmos (p ≥ 0.05)")

    def plot_results(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Boxplot de fitness
        fitness_data = [self.results['Adaptive PSO']['best_fitness'],
                       self.results['Adaptive PSO Local']['best_fitness']]

        ax1.boxplot(fitness_data, labels=['Adaptive PSO', 'Adaptive PSO Local'])
        ax1.set_ylabel('Fitness Final')
        ax1.set_title('Distribución del Fitness Final')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)

        # 2. Convergencia promedio
        for alg_name in self.results.keys():
            convergences = self.results[alg_name]['convergence']
            mean_convergence = np.mean(convergences, axis=0)
            std_convergence = np.std(convergences, axis=0)

            iterations = range(len(mean_convergence))
            ax2.plot(iterations, mean_convergence, label=alg_name, linewidth=2)
            ax2.fill_between(iterations,
                           mean_convergence - std_convergence,
                           mean_convergence + std_convergence,
                           alpha=0.2)

        ax2.set_xlabel('Iteraciones')
        ax2.set_ylabel('Fitness')
        ax2.set_title('Convergencia Promedio')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Histograma de fitness
        ax3.hist(self.results['Adaptive PSO']['best_fitness'], alpha=0.7,
                label='Adaptive PSO', bins=10, density=True)
        ax3.hist(self.results['Adaptive PSO Local']['best_fitness'], alpha=0.7,
                label='Adaptive PSO Local', bins=10, density=True)
        ax3.set_xlabel('Fitness Final')
        ax3.set_ylabel('Densidad')
        ax3.set_title('Distribución del Fitness Final')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Tiempos de ejecución
        time_data = [self.results['Adaptive PSO']['times'],
                    self.results['Adaptive PSO Local']['times']]

        ax4.boxplot(time_data, labels=['Adaptive PSO', 'Adaptive PSO Local'])
        ax4.set_ylabel('Tiempo (segundos)')
        ax4.set_title('Tiempos de Ejecución')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def create_summary_table(self):
        """Crear tabla resumen en formato DataFrame"""
        stats = self.calculate_statistics()

        data = []
        for alg_name, alg_stats in stats.items():
            data.append({
                'Algoritmo': alg_name,
                'Fitness Promedio': f"{alg_stats['mean_fitness']:.6f}",
                'Desv. Estándar': f"{alg_stats['std_fitness']:.6f}",
                'Mejor Fitness': f"{alg_stats['min_fitness']:.6f}",
                'Tiempo Promedio (s)': f"{alg_stats['mean_time']:.2f}",
                'Tasa Éxito (%)': f"{alg_stats['success_rate']:.1f}"
            })

        df = pd.DataFrame(data)
        print("\n📋 TABLA RESUMEN")
        print(df.to_string(index=False))

        return df

# --- Ejecución principal ---
def main():
    # Crear y ejecutar el entorno de pruebas
    test_env = PSOTestEnvironment(num_runs=10)

    # Ejecutar pruebas
    test_env.run_tests()

    # Mostrar estadísticas
    test_env.print_statistics()

    # Prueba estadística
    test_env.statistical_test()

    # Crear tabla resumen
    test_env.create_summary_table()

    # Generar gráficos
    test_env.plot_results()

    print("\nPruebas completadas exitosamente!")

if __name__ == "__main__":
    main()