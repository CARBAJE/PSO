import numpy as np
import math
from typing import Sequence

def rastrigin(X: Sequence[float], a: float = 10) -> float:
    n = len(X)
    return a * n + sum([(x**2 - a * np.cos(2 * math.pi * x)) for x in X])

def sphere(X: Sequence[float]) -> float:
    return sum([x**2 for x in X])