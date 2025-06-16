import time
import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
from particula import Particula
from matplotlib.animation import FuncAnimation

"Parametros Iniciales"
numParticulas = 20
iteraciones = 10
inercia = 0.8
aprendizajeLocal = 0.7 #aprendizajeL
aprendizajeGlobal = 1 #aprendizajeG

def f(x,y):
    "Funcion Objetivo"
    return x**2 + y**2 + (25*(np.sin(x) +np.sin(y)))

def imprimirLista(poblacion):
    # Obtener los atributos de cada objeto Particula como un diccionario
    data = [p.to_list() for p in poblacion]

    # Definir los encabezados de la tabla como una lista de strings
    headers = ["ID", "Posición", "Velocidad", "Inercia", "aprendizajeL", "aprendizajeG", "Mejor posición local"]

    # Imprimir la tabla utilizando tabulate
    print(tabulate(data, headers=headers, tablefmt="grid"))

def Posiciones(poblacion):
    posiciones = []
    for particula in poblacion:
        posiciones.append(particula.posicion)

    return np.array(posiciones)

def Velocidades(poblacion):
    velocidades = []
    for particula in poblacion:
        velocidades.append(particula.velocidad)

    return np.array(velocidades)

def crearPoblacion(numParticulas, inercia, aprendizajeL, aprendizajeG):
    np.random.seed(int(time.time()))
    particulas = []
    for i in range(numParticulas):
        posicion = np.random.uniform(-5, 5, 2)
        print(i)
        p = Particula(id=i,
                      posicion=posicion,
                      velocidad=np.random.randn(2)*0.1,
                      inercia=inercia,
                      b1=aprendizajeL,
                      b2=aprendizajeG,
                      mejorPos=posicion.copy())  # Usar .copy() para evitar referencias
        print(p.to_list())  # Cambiar para mostrar la lista en lugar del objeto
        particulas.append(p)
    return particulas

def mejorGlobal(poblacion):
    mejorG = []
    for elemento in poblacion:
        if len(mejorG) == 0:
            mejorG = elemento.posicion
        elif f(x=elemento.posicion[0], y=elemento.posicion[1]) < f(x=mejorG[0], y=mejorG[1]):
            mejorG = elemento.posicion

    return mejorG

def actualizarGeneracion(poblacion, mejorGlobal):
    r1, r2 = np.random.rand(2)
    for p in poblacion:
        p.actualizarVelocidad(r1, r2, mejorGlobal)
        p.actualizarPosicion()
        if(f(x=p.posicion[0], y=p.posicion[1]) < f(x=p.mejorPos[0], y=p.mejorPos[1])):
            p.mejorPos = p.posicion.copy()  # Usar .copy() para evitar referencias

    return poblacion

def PSO(numP, iteraciones, a, aLocal, aGlobal):
    poblacion = []
    mejorPosG = []
    fig, ax = plt.subplots(figsize=(8,6))
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    line, = ax.plot([], [], 'ro')  # Cambiado el estilo del marcador

    # Agregamos el fondo y las flechas
    x, y = np.meshgrid(np.linspace(-5,5,500), np.linspace(-5,5,500))
    z = f(x, y)
    img = ax.imshow(z, extent=[-5, 5, -5, 5], origin='lower', cmap='viridis', alpha=0.5)
    fig.colorbar(img, ax=ax)
    contours = ax.contour(x, y, z, 10, colors='black', alpha=0.4)
    ax.clabel(contours, inline=True, fontsize=8, fmt="%.0f")

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        titulo = f"Generacion {i} de la poblacion"
        print(titulo)

        nonlocal poblacion, mejorPosG
        if i > 0:
            poblacion = actualizarGeneracion(poblacion, mejorGlobal=mejorPosG)
        else:
            poblacion = crearPoblacion(numParticulas=numP, inercia=a, aprendizajeL=aLocal, aprendizajeG=aGlobal)

        mejorPosG = mejorGlobal(poblacion=poblacion)
        imprimirLista(poblacion)
        print(f"La mejor posicion global fue: {mejorPosG}")

        pos = Posiciones(poblacion)
        vels = Velocidades(poblacion)  # Agregamos este cálculo aquí
        line.set_data(pos[:,0], pos[:,1])

        # Actualizamos las flechas
        #flechas = ax.quiver(pos[:,0], pos[:,1], vels[:,0], vels[:,1], color='blue', width=0.005, angles='xy', scale_units='xy', scale=1)
        ax.set_title(titulo)
        return line,

    anim = FuncAnimation(fig, animate, frames=iteraciones, init_func=init, blit=True, repeat=False)

    # Opciones alternativas para guardar la animación
    try:
        # Opción 1: Intentar con pillow (más común y fácil de instalar)
        anim.save('PSO.gif', writer='pillow', fps=2)
        print("Animación guardada como PSO.gif usando pillow")
    except Exception as e:
        print(f"Error con pillow: {e}")
        try:
            # Opción 2: Usar ffmpeg si está disponible
            anim.save('PSO.mp4', writer='ffmpeg', fps=2)
            print("Animación guardada como PSO.mp4 usando ffmpeg")
        except Exception as e2:
            print(f"Error con ffmpeg: {e2}")
            # Opción 3: Solo mostrar la animación sin guardar
            print("No se pudo guardar la animación. Mostrando solo...")
            plt.show()

PSO(numP=numParticulas, iteraciones=iteraciones, a=inercia, aLocal=aprendizajeLocal, aGlobal=aprendizajeGlobal)