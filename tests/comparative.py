import numpy as np
import pandas as pd
import time
import math
from typing import List, Dict, Callable, Tuple, Optional, Sequence

from particle import *

def rastrigin(X: Sequence[float], a: float = 10) -> float:
    return a + sum((x**2 - a * np.cos(2 * math.pi * x)) for x in X)

def initializeParticles(particle_type: str, num_particle: int, dimensions: int, bounds, **kwargs) -> List:
    particles = []
    min_val = kwargs.get('min_val', 0.1)
    max_val = kwargs.get('max_val', math.e)
    n = kwargs.get('possible_val', 1000)

    bounds = np.array(bounds)
    range_val = bounds[:, 1] - bounds[:, 0]
    min_bound = bounds[:, 0]

    values = np.linspace(start=min_val, stop=max_val, num=n)

    weights = np.abs(np.sin(2 * values) + 0.5 * np.cos(3 * values))
    weights /= weights.sum()

    weights_speed = np.abs(0.7 * np.sin(math.e * values) + 0.8 * np.cos(math.pi * values))
    weights_speed /= weights_speed.sum()

    for i in range(num_particle):
        position = min_bound + range_val * np.random.beta(a=np.random.choice(values, p=weights), b=np.random.choice(values,p=(weights)), size=dimensions)

        speed_beta_a = np.random.choice(values, p=weights_speed)
        speed_beta_b = np.random.choice(values, p=weights_speed)
        speed = np.random.beta(a=speed_beta_a, b=speed_beta_b, size=dimensions)

        inertia = np.random.random() * 0.5 + 0.4

        if particle_type == 'global':
            particle = Particle(id=i+1, position=position, speed=speed, inertia=inertia, c1= kwargs.get('c1', 2), c2= kwargs.get('c2', 2))
        elif particle_type == 'local':
            particle = LocalParticle(id=i+1, position=position, speed=speed, inertia=inertia, c1= kwargs.get('c1', 2), c2= kwargs.get('c2', 2), num_neighbors=kwargs.get('num_neighbors', 2))
        elif particle_type == 'adaptive':
            particle = AdaptivePSOParticle(id=i+1, position=position, speed=speed)
        elif particle_type == 'adaptive_local':
            particle = AdaptivePSOLocal(id=i+1, position=position, speed=speed, num_neighbors=kwargs.get('num_neighbors', 2))
        else:
            raise ValueError(f"Tipo de partícula desconocido: {particle_type}")

        particles.append(particle)

    return particles

def setupNeighborhoods(particles: List, particle_type: str) -> None:
    if 'local' in particle_type:
        num_particle = len(particles)
        for particle in particles:
            particle.buildNeighborhood(particles, num_particle)
            if hasattr(particle, 'bestNeighbor'):
                particle.bestNeighbor = particle.position.copy()

def findGlobalBest(particles: List, f):
    global_best = particles[0].position.copy()
    global_best_fitness = f(global_best)

    for particle in particles[1:]:
        fitness = f(particle.bestPosition)
        if fitness < global_best_fitness:
            global_best_fitness = fitness
            global_best = particle.bestPosition.copy()

    return global_best, global_best_fitness

def updateParticles(particles: List, particle_type: str, global_best: List, dimensions: int, bounds, f):
    new_global_best = global_best.copy()
    global_best_fitness = f(global_best)

    bounds = np.array(bounds)

    for particle in particles:
        r1, r2 = np.random.rand(2)

        if particle_type == 'global':
            particle.updateSpeed(r1, r2, global_best)
            particle.updatePosition()

        elif particle_type == 'local':
            particle.findBestNeighbor(f)
            particle.updateSpeed(r1, r2)
            particle.updatePosition()

        elif particle_type == 'adaptive':
            particle.updateSpeed(r1, r2, global_best)
            particle.updatePosition()

        elif particle_type == 'adaptive_local':
            particle.findBestNeighbor(f)
            particle.updateSpeed(r1, r2)
            particle.updatePosition()

        particle.position = np.clip(particle.position, bounds[:, 0], bounds[:, 1])

        if particle.updateBestOwn(f):
            current_fitness = f(particle.bestPosition)
            if current_fitness < global_best_fitness:
                global_best_fitness = current_fitness
                new_global_best = particle.bestPosition.copy()

    return new_global_best, global_best_fitness

def findConvergenceIteration(fitness_history: List, threshold=1e-6):
    for i in range(1, len(fitness_history)):
        if abs(fitness_history[i] - fitness_history[i-1]) < threshold:
            return i
    return len(fitness_history) - 1

def calculateDiversity(particles: List) -> float:
    positions = np.array([p.position for p in particles])
    center = np.mean(positions, axis=0)
    distances = [np.linalg.norm(pos - center) for pos in positions]
    return np.mean(distances)

def runPsoAlgorithm(particle_type, func, bounds, num_particles=30, max_iterations=100, dimensions=2, num_neighbors=2):
    start_time = time.time()

    if isinstance(bounds, tuple):
        bounds_array = np.array([bounds] * dimensions)
    else:
        bounds_array = np.array(bounds)

    particles = initializeParticles(particle_type, num_particles, dimensions, bounds_array, num_neighbors=num_neighbors)

    setupNeighborhoods(particles, particle_type)

    global_best, global_best_fitness = findGlobalBest(particles, func)

    fitness_history = [global_best_fitness]

    for iteration in range(max_iterations):
        new_global_best, new_fitness = updateParticles(
            particles, particle_type, global_best, dimensions, bounds_array, func
        )

        if new_fitness < global_best_fitness:
            global_best_fitness = new_fitness
            global_best = new_global_best.copy()

        fitness_history.append(global_best_fitness)

    execution_time = time.time() - start_time

    result = {
        'algorithm': particle_type.replace('_', ' ').title() + ' PSO',
        'best_position': global_best,
        'best_fitness': global_best_fitness,
        'execution_time': execution_time,
        'fitness_history': fitness_history,
        'convergence_iteration': findConvergenceIteration(fitness_history),
        'final_diversity': calculateDiversity(particles)
    }

    if 'adaptive' in particle_type:
        avg_inertia = np.mean([p.inertia_mean for p in particles if hasattr(p, 'inertia_mean')])
        result['avg_inertia'] = avg_inertia

    return result


def comparePsoAlgorithms(test_functions, bounds=(-10, 10), num_runs=10, num_particles=30, max_iterations=100, dimensions=2):
    algorithms = ['global', 'local', 'adaptive', 'adaptive_local']

    all_results = []
    detailed_results = {}

    print("Iniciando comparación de algoritmos PSO...")
    print(f"Configuración: {num_particles} partículas, {max_iterations} iteraciones, {num_runs} runs")
    print("="*80)

    for func_name, func in test_functions.items():
        print(f"\nEvaluando función: {func_name}")
        print("-" * 40)

        for alg_type in algorithms:
            alg_name = alg_type.replace('_', ' ').title() + ' PSO'
            print(f"  Ejecutando {alg_name}...")

            run_results = []

            for run in range(num_runs):
                try:
                    result = runPsoAlgorithm(
                        particle_type=alg_type,
                        func=func,
                        bounds=bounds,
                        num_particles=num_particles,
                        max_iterations=max_iterations,
                        dimensions=dimensions,
                        num_neighbors=2
                    )
                    run_results.append(result)

                except Exception as e:
                    print(f"    Error en run {run+1}: {e}")
                    continue

            if run_results:
                best_fitnesses = [r['best_fitness'] for r in run_results]
                execution_times = [r['execution_time'] for r in run_results]
                convergence_iterations = [r['convergence_iteration'] for r in run_results]
                diversities = [r['final_diversity'] for r in run_results]

                result_summary = {
                    'Function': func_name,
                    'Algorithm': alg_name,
                    'Best_Fitness_Mean': np.mean(best_fitnesses),
                    'Best_Fitness_Std': np.std(best_fitnesses),
                    'Best_Fitness_Min': np.min(best_fitnesses),
                    'Best_Fitness_Max': np.max(best_fitnesses),
                    'Execution_Time_Mean': np.mean(execution_times),
                    'Execution_Time_Std': np.std(execution_times),
                    'Convergence_Iteration_Mean': np.mean(convergence_iterations),
                    'Convergence_Iteration_Std': np.std(convergence_iterations),
                    'Diversity_Mean': np.mean(diversities),
                    'Success_Rate': len([f for f in best_fitnesses if f < 1e-6]) / len(best_fitnesses),
                    'Runs_Completed': len(run_results)
                }

                if 'adaptive' in alg_type:
                    avg_inertias = [r.get('avg_inertia', 0) for r in run_results if 'avg_inertia' in r]
                    if avg_inertias:
                        result_summary['Avg_Inertia_Mean'] = np.mean(avg_inertias)
                        result_summary['Avg_Inertia_Std'] = np.std(avg_inertias)

                all_results.append(result_summary)
                detailed_results[f"{func_name}_{alg_name}"] = run_results

                print(f"    Completado: {len(run_results)} runs exitosos")
                print(f"    Mejor fitness: {np.min(best_fitnesses):.6e}")

    return pd.DataFrame(all_results), detailed_results

def printSummaryStatistics(results_df):
    """Imprime estadísticas resumidas"""
    print("\n" + "="*80)
    print("RESULTADOS DETALLADOS")
    print("="*80)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.6e}'.format)

    print(results_df.to_string(index=False))

def printAlgorithmRanking(results_df):
    print("\n" + "="*50)
    print("RANKING POR FUNCIÓN")
    print("="*50)

    for func in results_df['Function'].unique():
        print(f"\n{func}:")
        func_results = results_df[results_df['Function'] == func].copy()
        func_results = func_results.sort_values('Best_Fitness_Mean')

        for i, (_, row) in enumerate(func_results.iterrows(), 1):
            print(f"  {i}. {row['Algorithm']}: {row['Best_Fitness_Mean']:.6e}\t"f"(±{row['Best_Fitness_Std']:.6e})")

def printOverallPerformance(results_df):
    print("\n" + "="*50)
    print("RENDIMIENTO GENERAL POR ALGORITMO")
    print("="*50)

    summary = results_df.groupby('Algorithm').agg({
        'Best_Fitness_Mean': ['mean', 'std', 'min'],
        'Execution_Time_Mean': ['mean', 'std'],
        'Success_Rate': 'mean',
        'Convergence_Iteration_Mean': 'mean'
    }).round(6)

    print(summary)

if __name__ == "__main__":
    test_functions = {
        'Rastrigin': rastrigin,
        }

    results_df, detailed_results = comparePsoAlgorithms(
        test_functions=test_functions,
        bounds=(-10, 10),
        num_runs=5,  # Reducido para pruebas rápidas
        num_particles=20,
        max_iterations=50,
        dimensions=2
    )

    printSummaryStatistics(results_df)
    printAlgorithmRanking(results_df)
    printOverallPerformance(results_df)

    # results_df.to_csv('pso_comparison_results.csv', index=False)
    print(f"\nComparación completada. Total de experimentos: {len(results_df)}")