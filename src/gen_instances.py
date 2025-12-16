import numpy as np
from .TSP import TSP_Instance
import os
from .settings import instance_path
import pickle

def save_instances(filename, instances: list[TSP_Instance]):
    os.makedirs(instance_path, exist_ok=True)

    points = []
    for instance in instances:
        points.append(instance.city_locations)

    with open(instance_path + filename, "wb") as f:
        pickle.dump(np.array(points), f)

def read_instances(filename):
    with open(instance_path + filename, "rb") as f:
        points = pickle.load(f)

    instances = []
    for instance_points in points:
        instance = TSP_Instance(instance_points)
        instances.append(instance)
    
    return instances

def generate_instances(filename, instance_count=1, cities=20, seed=42):
    np.random.seed(seed)
    dim = 2  # Dimensión para las coordenadas de la ciudad (2D: x, y)

    instances = []
    for _ in range(instance_count):
        city_points = np.random.rand(cities, dim)  # Generar puntos aleatorios para las ciudades
        instances.append(TSP_Instance(city_points))

    save_instances(filename, instances)
    return instances