import numpy as np
from .TSP import TSP_Instance, TSP_State
import os
from .solvers.ortools import solve
from .settings import instance_path, data_path
import pickle

def save_instances(filename, instances: list[TSP_Instance]):
    os.makedirs(instance_path, exist_ok=True)

    points = []
    for instance in instances:
        points.append(instance.city_locations)

    with open(instance_path + filename, "wb") as f:
        pickle.dump(np.array(points), f)

def read_instances(filename) -> list[TSP_Instance]:
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

def generate_train_data(instance_file, data_filename, size=10000):
    instances = read_instances(instance_file)
    X_src_all = []
    visited_all = []
    Y_all = []

    for instance in instances:
        solution = solve(instance)
        state = TSP_State(instance)

        for city in solution.tour[1:-1]:
            X_src = np.array(instance.city_locations)
            visited = np.pad(np.array(state.tour), (0, instance.num_cities - len(state.tour)), 'constant', constant_values=-1)
            Y = np.zeros(instance.num_cities)
            Y[city] = 1  # One-hot encoding de la ciudad visitada

            X_src_all.append(X_src)
            visited_all.append(visited)
            Y_all.append(Y)
            state.visit_city(city)

            if len(X_src_all) >= size:
                break
        
        if len(X_src_all) >= size:
            break
    
    save_train_data(data_filename, np.array(X_src_all), np.array(visited_all), np.array(Y_all))

def save_train_data(filename, X_src_all, visited_all, Y_all):
    os.makedirs(data_path, exist_ok=True)

    with open(data_path + filename, "wb") as f:
        pickle.dump((X_src_all, visited_all, Y_all), f)

def read_train_data(filename):
    with open(data_path + filename, "rb") as f:
        X_src_all, visited_all, Y_all = pickle.load(f)
    return X_src_all, visited_all, Y_all