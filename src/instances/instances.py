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
        pickle.dump(np.array(points), f)

def read_instances(filename) -> list[TSPInstance]:
    with open(INSTANCE_FOLDER / filename, "rb") as f:
        points = pickle.load(f)

    instances = []
    for instance_points in points:
        instance = TSPInstance(instance_points)
        instances.append(instance)
    
    return instances

def generate_instances(filename, instance_count=1, cities=20, seed=42):
    np.random.seed(seed)
    dim = 2  # Dimensión para las coordenadas de la ciudad (2D: x, y)

    instances = []
    for _ in range(instance_count):
        city_points = np.random.rand(cities, dim)  # Generar puntos aleatorios para las ciudades
        instances.append(TSPInstance(city_points))

    save_instances(filename, instances)
    return instances