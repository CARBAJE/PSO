import numpy as np

class FocalPSO:
    def _init_(self, func, dim, num_particles=30, max_iter=100,
                 w=0.7, c1=1.5, c2=1.5, neigh_size=3,
                 bounds=None, seed=None):
        """
        func       : función objetivo, recibe array de tamaño dim y devuelve escalar
        dim        : número de variables
        num_particles : tamaño del enjambre
        max_iter   : número máximo de iteraciones
        w, c1, c2  : coeficientes de inercia y aceleración
        neigh_size : tamaño de cada vecindario (incluye partícula focal)
        bounds     : lista de tuplas (min, max) para cada dimensión
        seed       : semilla para reproducibilidad
        """
        self.func = func
        self.dim = dim
        self.Np = num_particles
        self.max_iter = max_iter
        self.w, self.c1, self.c2 = w, c1, c2
        self.neigh_size = neigh_size
        self.bounds = bounds if bounds is not None else [(-1,1)]*dim

        if seed is not None:
            np.random.seed(seed)

        # 1. Generación de posiciones y velocidades
        self.X = np.array([np.random.uniform(low, high, num_particles)
                           for (low, high) in self.bounds]).T  # shape (Np, dim)
        self.V = np.zeros((self.Np, self.dim))

        # 4–5. Mejor posición individual (pbest) y su valor
        self.pbest = self.X.copy()
        self.fpbest = np.array([self.func(x) for x in self.X])

        # Matriz de vecinos (topología focal)
        self._build_focal_neighborhood()

    def _build_focal_neighborhood(self):
        # Cada partícula i tiene un vecindario focal: ella misma + neigh_size-1 aleatorias
        self.neighbors = np.zeros((self.Np, self.neigh_size), dtype=int)
        for i in range(self.Np):
            # incluimos siempre i, luego seleccionamos aleatoriamente neigh_size-1 sin reemplazo
            others = [j for j in range(self.Np) if j != i]
            picks = np.random.choice(others, self.neigh_size-1, replace=False)
            self.neighbors[i, 0] = i
            self.neighbors[i, 1:] = picks

    def optimize(self):
        for t in range(self.max_iter):
            # 6. Para cada partícula, encontrar líder del vecindario
            leaders = np.zeros(self.Np, dtype=int)
            for i in range(self.Np):
                idxs = self.neighbors[i]
                # seleccionamos la mejor fpbest en ese vecindario
                local_best_idx = idxs[np.argmin(self.fpbest[idxs])]
                leaders[i] = local_best_idx

            # 7. Actualizar velocidad y posición
            r1, r2 = np.random.rand(self.Np, self.dim), np.random.rand(self.Np, self.dim)
            Xl = self.pbest[leaders]    # posiciones de los líderes
            self.V = ( self.w * self.V
                      + self.c1 * r1 * (self.pbest - self.X)
                      + self.c2 * r2 * (Xl     - self.X) )
            self.X = self.X + self.V

            # 8. Aplicar restricciones de dominio
            for d in range(self.dim):
                low, high = self.bounds[d]
                self.X[:,d] = np.clip(self.X[:,d], low, high)

            # 9–10. Evaluar y actualizar pbest
            fvals = np.array([self.func(x) for x in self.X])
            better = fvals < self.fpbest
            self.pbest[better] = self.X[better]
            self.fpbest[better] = fvals[better]

        # Devolver mejor solución encontrada
        best_idx = np.argmin(self.fpbest)
        return self.pbest[best_idx], self.fpbest[best_idx]


def langermann_function(x, m=5, c=None, A=None):
    """
    Función de Langermann para optimización.

    Parámetros:
    x : array de entrada (usualmente 2D)
    m : número de términos en la suma (por defecto 5)
    c : coeficientes c_i (por defecto valores estándar)
    A : matriz A de coeficientes (por defecto valores estándar para 2D)

    Dominio típico: [0, 10]^d
    Mínimo global conocido para 2D: aproximadamente -5.1621 en (2.00299219, 1.006096)
    """
    if c is None:
        c = np.array([1, 2, 5, 2, 3])  # coeficientes estándar

    if A is None:
        # Matriz A estándar para la función de Langermann 2D
        A = np.array([
            [3, 5],
            [5, 2],
            [2, 1],
            [1, 4],
            [7, 9]
        ])

    x = np.array(x)
    result = 0.0

    for i in range(m):
        # Calcular la distancia euclidiana al cuadrado
        dist_sq = np.sum((x - A[i])**2)

        # Calcular el término de la suma
        term = c[i] * np.exp(-dist_sq / np.pi) * np.cos(np.pi * dist_sq)
        result += term

    return -result  # Negativo porque queremos maximizar la función original


# --- Ejemplo de uso para Langermann ---
if __name__ == "_main_":
    print("Optimizando la función de Langermann con PSO Focal")
    print("=" * 50)

    # Configurar el PSO para Langermann
    pso = FocalPSO(
        func=langermann_function,
        dim=2,
        num_particles=100,  # Más partículas para función compleja
        max_iter=500,       # Más iteraciones
        w=0.4,             # Inercia menor para mejor exploración local
        c1=1.8,            # Coeficiente cognitivo
        c2=1.8,            # Coeficiente social
        neigh_size=5,      # Vecindario más grande
        bounds=[(0, 10), (0, 10)],  # Dominio estándar de Langermann
        seed=42
    )

    # Ejecutar optimización
    best_x, best_f = pso.optimize()

    print(f"Mejor posición encontrada: [{best_x[0]:.6f}, {best_x[1]:.6f}]")
    print(f"Mejor valor de función: {best_f:.6f}")
    print(f"Valor de Langermann original: {-best_f:.6f}")
    print()
    print("Nota: El mínimo global teórico está cerca de:")
    print("Posición: [2.00299219, 1.006096]")
    print("Valor: -5.1621")

    # Verificar el valor en el óptimo teórico
    theoretical_optimum = np.array([2.00299219, 1.006096])
    theoretical_value = langermann_function(theoretical_optimum)
    print(f"Valor teórico verificado: {theoretical_value:.6f}")
    print(f"Langermann teórico: {-theoretical_value:.6f}")
