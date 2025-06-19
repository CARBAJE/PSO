# PSO - Particle Swarm Optimization

Este proyecto implementa y visualiza variantes del algoritmo de Optimización por Enjambre de Partículas (PSO) en Python, incluyendo versiones global, local, adaptativa y adaptativa local. Incluye herramientas para análisis comparativo, visualización estática y animada, y una interfaz interactiva en Jupyter Notebook.

## Estructura del Proyecto

- `pso.py`: Lógica principal del algoritmo PSO y variantes.
- `particles.py`: Definición de las clases de partículas (global, local, adaptativa).
- `objective_functions.py`: Funciones objetivo (Rastrigin, Sphere, etc.).
- `comparative.py`: Script para comparar el rendimiento de los algoritmos PSO.
- `visualizer.ipynb`: Notebook interactivo para visualización y experimentación.
- `tests/`: Scripts de prueba y experimentación.

## Requisitos

- Python 3.8+
- Paquetes:
  - numpy
  - pandas
  - matplotlib
  - networkx
  - ipywidgets
  - jupyter
  - ffmpeg (para animaciones MP4)

Puedes instalar los paquetes necesarios con:

```bash
pip install numpy pandas matplotlib networkx ipywidgets jupyter
```

Para exportar animaciones a MP4, asegúrate de tener `ffmpeg` instalado y accesible en tu PATH.

## Uso

### 1. Visualización Interactiva

Abre el notebook `visualizer.ipynb` en Jupyter y ejecuta las celdas para:
- Visualizar distribuciones de inicialización.
- Ver la función objetivo.
- Analizar la topología de vecindario.
- Ejecutar y comparar variantes de PSO.
- Generar animaciones del proceso de optimización.

### 2. Comparación de Algoritmos

Ejecuta el script `comparative.py` para comparar el rendimiento de las variantes de PSO sobre diferentes funciones objetivo. Los resultados se muestran en consola y pueden exportarse a DataFrame de pandas.

```bash
python comparative.py
```

### 3. Personalización

Puedes modificar los parámetros de PSO (número de partículas, iteraciones, vecinos, etc.) directamente en los scripts o el notebook.

## Créditos

Desarrollado por José Emiliano Carrillo Barreiro.
Inspirado en la literatura clásica de PSO y variantes adaptativas.
