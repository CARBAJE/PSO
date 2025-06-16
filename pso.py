# pso.py

import numpy as np
import time
import math
from typing import List, Callable
from particles import *

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

def findGlobalBest(particles: List, f: Callable):
    global_best = particles[0].position.copy()
    global_best_fitness = f(global_best)
    for particle in particles[1:]:
        fitness = f(particle.bestPosition)
        if fitness < global_best_fitness:
            global_best_fitness = fitness
            global_best = particle.bestPosition.copy()
    return global_best, global_best_fitness

def updateParticles(particles: List, particle_type: str, global_best: np.ndarray, bounds: np.ndarray, f: Callable):
    new_global_best = global_best.copy()
    global_best_fitness = f(global_best)

    for particle in particles:
        r1, r2 = np.random.rand(2)
        if particle_type == 'global':
            particle.updateSpeed(r1, r2, global_best)
        elif particle_type == 'local':
            particle.findBestNeighbor(f)
            particle.updateSpeed(r1, r2)
        elif particle_type == 'adaptive':
            particle.updateSpeed(r1, r2, global_best)
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

def runPsoAlgorithm(particle_type: str, func: Callable, bounds: tuple, num_particles: int, max_iterations: int, dimensions: int, **kwargs):
    start_time = time.time()

    bounds_array = np.array([bounds] * dimensions)

    particles = initializeParticles(particle_type, num_particles, dimensions, bounds_array, **kwargs)
    setupNeighborhoods(particles, particle_type)
    global_best, global_best_fitness = findGlobalBest(particles, func)
    fitness_history = [global_best_fitness]

    for _ in range(max_iterations):
        new_global_best, new_fitness = updateParticles(particles, particle_type, global_best, bounds_array, func)
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