#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Markov Chain: Metropolis-Hastings
import numpy as np
import matplotlib.pyplot as plt
# Nueva matriz de transición
transition_matrix = np.array([
    [0.7, 0.1, 0.1, 0.1],
    [0.2, 0.6, 0.1, 0.1],
    [0.1, 0.2, 0.5, 0.2],
    [0.05, 0.15, 0.2, 0.6]
])

# Distribución inicial (4 estados)
initial_distribution = np.array([0.25, 0.25, 0.25, 0.25])

# Listas para guardar la evolución de la distribución y la iteración
distributions = [initial_distribution]
iterations = 50

# Iterar para calcular la distribución en cada paso
for i in range(iterations):
    new_distribution = np.dot(distributions[-1], transition_matrix)
    distributions.append(new_distribution)

# Convertir a un array de NumPy para facilitar el acceso
distributions = np.array(distributions)

# Crear la visualización
plt.figure(figsize=(12, 8))

for i in range(4):
    plt.plot(range(iterations + 1), distributions[:, i], label=f"Estado {i+1}")

plt.title("Convergencia a la Distribución Estacionaria")
plt.xlabel("Iteraciones")
plt.ylabel("Probabilidad")
plt.legend()
plt.grid(True)
plt.show()


# In[2]:


import networkx as nx
import matplotlib.pyplot as plt

# Matriz de transición (usando la nueva matriz que proporcionaste)
transition_matrix = np.array([
    [0.7, 0.1, 0.1, 0.1],
    [0.2, 0.6, 0.1, 0.1],
    [0.1, 0.2, 0.5, 0.2],
    [0.05, 0.15, 0.2, 0.6]
])

# Crear un grafo dirigido desde la matriz de transición
G = nx.DiGraph(transition_matrix)

# Obtener las clases de comunicación
communication_classes = list(nx.strongly_connected_components(G))

# Mostrar el grafo y las clases de comunicación
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, arrowsize=20, node_size=700, node_color="skyblue")
plt.title("Grafo de la Cadena de Markov")
plt.show()

print("Clases de Comunicación:")
for i, cc in enumerate(communication_classes):
    print(f"Clase {i + 1}: {cc}")

# Verificar aperiodicidad e irreducibilidad
is_aperiodic = nx.is_aperiodic(G)
is_irreducible = nx.is_strongly_connected(G)

print("\nAperiodicidad:", is_aperiodic)
print("Irreducibilidad:", is_irreducible)


# In[3]:


R_x = np.array([100, -500, 25, 1])  
E_Rx = R_x @ distributions[-1,:]
E_Rx



# In[4]:


aprox_dist_estac=distributions[-1,:]
aprox_dist_estac


# In[12]:


R=np.array([10.0,-50.0,3.0,0])

rec_por_pi=R@aprox_dist_estac
R[-1]=-rec_por_pi/aprox_dist_estac[-1]
R@aprox_dist_estac


# In[16]:


# aceptacion o rechazo 
import numpy as np
import scipy.stats as st
xmin, xmax = -2, 2
x = np.linspace(xmin, xmax, 200)
y = st.norm(0, 1).pdf(x)

import plotly.express as px

M=max(y)+0.01


import plotly.graph_objs as go

fig=go.Figure()
traces=[go.Scatter(x=np.ones(100)*xmin,y=np.linspace(0,M,100),mode='lines',marker=dict(color='black')),
        go.Scatter(x=np.ones(100)*xmax,y=np.linspace(0,M,100),mode='lines',marker=dict(color='black')),
        go.Scatter(x=x,y=np.ones(len(x))*M,mode='lines',marker=dict(color='black')),
        go.Scatter(x=x,y=y,mode='lines'),]
[fig.add_trace(trace) for trace in traces]
fig.update_layout(
    title='Aceptación rechazo para la normal truncada',xaxis_title='x', yaxis_title='f(x)',
    height=500, width=500,
    xaxis=dict(range=[xmin-0.1, xmax+0.1]),  showlegend=False
)


fig.show()


# In[17]:


# Generamos  N puntos dentro del cuadrado y vemos cuantos aceptamos y rechazamos
N_puntos=1000
p_generados_x=np.random.random(N_puntos)*(xmax-xmin)+xmin
p_generados_y=np.random.random(size=N_puntos)*M

f_evaluando_x=st.norm(0,1).pdf(p_generados_x)
idx_aceptados=p_generados_y<f_evaluando_x
idx_rechazados=p_generados_y>=f_evaluando_x

fig=go.Figure()
traces=[go.Scatter(x=np.ones(100)*xmin,y=np.linspace(0,M,100),mode='lines',marker=dict(color='black')),
        go.Scatter(x=np.ones(100)*xmax,y=np.linspace(0,M,100),mode='lines',marker=dict(color='black')),
        go.Scatter(x=x,y=np.ones(len(x))*M,mode='lines',marker=dict(color='black')),
        go.Scatter(x=x,y=y,mode='lines'),
        go.Scatter(x=p_generados_x[idx_aceptados],y=p_generados_y[idx_aceptados],mode='markers'),
        go.Scatter(x=p_generados_x[idx_rechazados],y=p_generados_y[idx_rechazados],mode='markers')]
[fig.add_trace(trace) for trace in traces]
fig.update_layout(
    title='Aceptación rechazo para la normal truncada',xaxis_title='x', yaxis_title='f(x)',
    height=500, width=500,
    xaxis=dict(range=[xmin-0.1, xmax+0.1]),  showlegend=False
)


# In[18]:


import matplotlib.pyplot as plt

fig=px.histogram(p_generados_x[idx_aceptados])
fig.update_layout(
    title='Histograma de puntos aceptados para una Gaussiana',xaxis_title='x', yaxis_title='f(x)',
    height=500, width=500,showlegend=False
)
fig.show()


# In[19]:


import numpy as np
from scipy.stats import beta, norm

def metropolis_hastings(distribucion_objetivo, distribucion_propuesta, parametros_propuesta, estado_inicial, n_iteraciones):
    """
    Algoritmo de Metropolis-Hastings para generar muestras de una distribución objetivo.

    :param distribucion_objetivo: función de la distribución objetivo que queremos muestrear
    :param distribucion_propuesta: función de la distribución de propuestas para explorar el espacio de estados
    :param parametros_propuesta: parámetros para la distribución de propuestas
    :param estado_inicial: valor inicial de la cadena de Markov
    :param n_iteraciones: número de iteraciones del algoritmo
    :return: muestras generadas por el algoritmo
    """
    muestras = [estado_inicial]
    estado_actual = estado_inicial

    for _ in range(n_iteraciones):
        # Genera una nueva muestra desde la distribución de propuestas
        estado_propuesto = distribucion_propuesta(estado_actual, **parametros_propuesta)
        # Calcula la razón de aceptación (Aceptación de Metropolis)
        razon_aceptacion = min(1, distribucion_objetivo(estado_propuesto) / distribucion_objetivo(estado_actual))
        # Acepta o rechaza la nueva muestra con la probabilidad de la razón de aceptación
        if np.random.rand() < razon_aceptacion:
            estado_actual = estado_propuesto

        muestras.append(estado_actual)

    return muestras

# Definimos nuestra distribución objetivo y de propuestas
def distribucion_objetivo_beta(x):
    # Beta con parámetros alpha=2, beta=5
    return beta.pdf(x, 2, 5)

def distribucion_propuesta_normal(estado_actual, mu, sigma):
    # Normal centrada en el estado actual
    return norm.rvs(loc=estado_actual, scale=sigma)

# Ejemplo de uso de la función metropolis_hastings
np.random.seed(0)  # Semilla para reproducibilidad
muestras_mh = metropolis_hastings(
    distribucion_objetivo=distribucion_objetivo_beta, 
    distribucion_propuesta=distribucion_propuesta_normal, 
    parametros_propuesta={'mu': 0, 'sigma': .1}, 
    estado_inicial=0.1, 
    n_iteraciones=1000
)

# Visualización de las muestras generadas
plt.figure(figsize=(10, 5))
plt.plot(muestras_mh, '-o', markersize=4, alpha=0.6)
plt.title('Muestras Generadas por el Algoritmo de Metropolis-Hastings')
plt.xlabel('Iteración')
plt.ylabel('Valor')
plt.show()


# In[20]:


import plotly.express as px
import seaborn as sns

sns.distplot(muestras_mh[:],label='distribución generada');
plt.plot(np.linspace(0,1,1000),distribucion_objetivo_beta(np.linspace(0,1,1000)),label='distribución real');
plt.legend();


# In[ ]:




