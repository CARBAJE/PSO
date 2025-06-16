import pandas as pd
import numpy as np

from pso import runPsoAlgorithm
from objective_functions import rastrigin, sphere

def comparePsoAlgorithms(test_functions, bounds_map, num_runs=10,
                          num_particles=30, max_iterations=100, dimensions=2):
    algorithms = ['global', 'local', 'adaptive', 'adaptive_local']
    all_results = []

    print("Iniciando comparación de algoritmos PSO...")
    print(f"Configuración: {num_particles} partículas, {max_iterations} iteraciones, {num_runs} runs")
    print("="*80)

    for func_name, func in test_functions.items():
        print(f"\nEvaluando función: {func_name}")
        print("-" * 40)
        bounds = bounds_map[func_name]

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
                        num_neighbors=4  # Puedes ajustar esto o pasarlo como parámetro
                    )
                    run_results.append(result)
                except Exception as e:
                    print(f"    Error en run {run+1}: {e}")
                    import traceback
                    traceback.print_exc()
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
                    'Execution_Time_Mean': np.mean(execution_times),
                    'Convergence_Iteration_Mean': np.mean(convergence_iterations),
                    'Diversity_Mean': np.mean(diversities),
                    'Success_Rate': len([f for f in best_fitnesses if abs(f) < 1e-4]) / len(best_fitnesses),
                    'Runs_Completed': len(run_results)
                }

                if 'adaptive' in alg_type:
                    avg_inertias = [r.get('avg_inertia', 0) for r in run_results if 'avg_inertia' in r]
                    if avg_inertias:
                        result_summary['Avg_Inertia_Mean'] = np.mean(avg_inertias)

                all_results.append(result_summary)
                print(f"    Completado: {len(run_results)} runs exitosos. Mejor fitness (Min): {np.min(best_fitnesses):.6e}")

    return pd.DataFrame(all_results)

def printSummaryStatistics(results_df):
    if results_df.empty:
        print("\nNo se completaron ejecuciones exitosas.")
        return
    print("\n" + "="*80 + "\nRESULTADOS DETALLADOS\n" + "="*80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.6e}'.format)
    print(results_df.to_string(index=False))

def printAlgorithmRanking(results_df):
    if results_df.empty:
        return
    print("\n" + "="*50 + "\nRANKING POR FUNCIÓN (basado en Media de Mejor Fitness)\n" + "="*50)
    for func in results_df['Function'].unique():
        print(f"\nFunción: {func}")
        func_results = results_df[results_df['Function'] == func].copy().sort_values('Best_Fitness_Mean')
        for i, (_, row) in enumerate(func_results.iterrows(), 1):
            print(f"  {i}. {row['Algorithm']:<20}: {row['Best_Fitness_Mean']:.6e} (±{row['Best_Fitness_Std']:.6e})")

if __name__ == "__main__":
    test_functions = {
        'Rastrigin': rastrigin
    }

    bounds_map = {
        'Rastrigin': (-5.12, 5.12)
    }

    # Ejecuta la comparación
    results_df = comparePsoAlgorithms(
        test_functions=test_functions,
        bounds_map=bounds_map,
        num_runs=5,
        num_particles=30,
        max_iterations=100,
        dimensions=2
    )

    printSummaryStatistics(results_df)
    printAlgorithmRanking(results_df)