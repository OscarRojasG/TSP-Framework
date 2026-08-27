# TSP-Framework
**Un entorno de experimentación en Deep Learning para el Problema del Vendedor Viajero (TSP)**

Este repositorio contiene un framework modular diseñado para investigar, entrenar y evaluar redes neuronales aplicadas al *Traveling Salesperson Problem* (TSP). 

El proyecto está construido bajo una filosofía educativa y experimental, dividiendo las distintas técnicas del estado del arte en una serie de **10 notebooks interactivos**. Estos notebooks funcionan como guías paso a paso para entender desde las arquitecturas base hasta algoritmos avanzados de Reinforcement Learning y Graph Neural Networks.

## Guías y Tutoriales (Notebooks)

El corazón de este framework se encuentra en los siguientes notebooks, diseñados para ser explorados de forma secuencial o independiente:

1. `cost_prediction.ipynb`
   Predicción del costo óptimo. Un primer acercamiento a la arquitectura Transformer resolviendo un problema de regresión más sencillo (estimar la distancia final del tour) para familiarizarse con los embeddings y el procesamiento del encoder antes de generar secuencias.

2. `glimpse.ipynb`
   Análisis del mecanismo **Glimpse** (Cross-Attention iterativo). Cómo permitir que el decodificador "eche un vistazo" extra a la memoria del encoder antes de decidir la próxima ciudad.

3. `architecture_variants.ipynb`
   Exploración de las arquitecturas base (Encoder-Decoder basados en Transformers). Se analizan distintas formas de procesar el contexto y representar el estado del tour.

4. `positional_encoding.ipynb`
   Estrategias de codificación posicional. Implementación de **Spatial Positional Encoding** (para inyectar consciencia 2D nativa) y **Circular Positional Encoding** (para secuencias cerradas).

5. `sparse_attention.ipynb`
   Mecanismos de **Atención Esparsa**. Estrategias como K-Nearest Neighbors (KNN) o triangulación de Delaunay para forzar a la red a enfocarse solo en vecindarios lógicos locales.

6. `linear_attention.ipynb`
   Aproximaciones de **Atención Lineal**. Técnicas para mitigar el cuello de botella de memoria $O(N^2)$ de la atención clásica y escalar el modelo a miles de ciudades.

7. `gat.ipynb`
   Integración de **Graph Attention Networks (GAT)**. Cómo reemplazar o combinar el Transformer tradicional con una GNN pura para aprovechar la topología esparsa del grafo.

8. `augmentation.ipynb`
   Técnicas de *Data Augmentation* espacial para grafos (rotaciones, traslaciones, simetrías) para evitar el sobreajuste y mejorar la generalización del modelo en mapas no vistos.

9. `exit.ipynb`
   Implementación de **Expert Iteration (ExIt)**. Un pipeline de entrenamiento avanzado donde el modelo aprende de sí mismo mediante un ciclo iterativo de búsqueda guiada (*Lookahead + Rollouts*) y destilación de políticas.

10. `reinforce.ipynb`
   Entrenamiento mediante Aprendizaje por Refuerzo puro. Implementación del algoritmo **REINFORCE (Policy Gradient)** con *baseline* de validación para entrenar sin necesidad de soluciones óptimas pre-calculadas.

---

## Instalación

Para ejecutar el framework y los notebooks en tu máquina local, sigue estos pasos:

**1. Clonar el repositorio**
```bash
git clone [https://github.com/OscarRojasG/TSP-Framework.git](https://github.com/OscarRojasG/TSP-Framework.git)
cd TSP-Framework
```

**2. Crear y activar un entorno virtual (Recomendado)**
Para mantener las dependencias aisladas, crea un entorno virtual de Python:
```bash
# En sistemas Linux/macOS:
python3 -m venv venv
source venv/bin/activate

# En Windows:
python -m venv venv
venv\Scripts\activate
```

**3. Instalar dependencias**
Con el entorno activado, instala los paquetes requeridos usando `pip`:
```bash
pip install -r requirements.txt
```