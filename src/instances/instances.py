from TSP import TSPInstance
import numpy as np
import os
import pickle
from settings import INSTANCE_FOLDER

def save_instances(filename, instances: list[TSPInstance]):
    os.makedirs(INSTANCE_FOLDER, exist_ok=True)

    points = []
    for instance in instances:
        points.append(instance.city_locations)

    with open(INSTANCE_FOLDER / filename, "wb") as f:
        pickle.dump(points, f)

def read_instances(filename) -> list[TSPInstance]:
    with open(INSTANCE_FOLDER / filename, "rb") as f:
        points = pickle.load(f)

    instances = []
    for instance_points in points:
        instance = TSPInstance(instance_points)
        instances.append(instance)
    
    return instances

def generate_instances(filename, instance_count=1, cities=20, seed=42):
    """
    Genera un archivo con instancias del TSP. 
    'cities' puede ser un entero (ej. 20) o un iterable de enteros (ej. range(5, 51)).
    Si es un iterable, instance_count se dividirá equitativamente entre los diferentes tamaños.
    """
    np.random.seed(seed)
    dim = 2  # Dimensión para las coordenadas de la ciudad (2D: x, y)

    # Convertir 'cities' a una lista si es un número entero
    if isinstance(cities, int):
        sizes_list = [cities]
    else:
        sizes_list = list(cities)

    num_sizes = len(sizes_list)
    
    # Calcular distribución base y el resto (por si la división no es exacta)
    base_count = instance_count // num_sizes
    remainder = instance_count % num_sizes

    instances = []
    
    for i, city_size in enumerate(sizes_list):
        # A los primeros 'remainder' tamaños se les asigna una instancia extra
        count_for_this_size = base_count + (1 if i < remainder else 0)
        
        for _ in range(count_for_this_size):
            city_points = np.random.rand(city_size, dim)  # Generar puntos aleatorios
            instances.append(TSPInstance(city_points))

    save_instances(filename, instances)
    return instances