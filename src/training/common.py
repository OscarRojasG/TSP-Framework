import os
import torch
import json
import random
import matplotlib.pyplot as plt
from dataclasses import dataclass
from settings import MODELS_FOLDER, HYPERPARAMETERS_FOLDER, EXPERIMENTS_FOLDER

@dataclass
class LRConfig:
    value: float            # Tasa de aprendizaje inicial
    factor: float = 0.5     # Factor de reducción
    patience: int = 999999  # Épocas sin mejora antes de reducir el LR
    min: float = 0.0        # Tasa de aprendizaje mínima permitida

def config_training(model, seed):
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() 
                          else "mps" if torch.backends.mps.is_available() 
                          else "cpu")
    print(f"** Usando dispositivo: {device}")
    torch.set_num_threads(os.cpu_count())
    return model.to(device), device

def save_model(model, model_name, verbose=True):
    os.makedirs(HYPERPARAMETERS_FOLDER, exist_ok=True)
    with open(str(HYPERPARAMETERS_FOLDER / model_name) + ".json", 'w') as f:
        json.dump(model.hyperparams, f, indent=4)

    os.makedirs(MODELS_FOLDER, exist_ok=True)
    weights = model.state_dict()
    torch.save(weights, str(MODELS_FOLDER / model_name) + ".pth")
    if verbose:
        print(f"** Modelo guardado en {MODELS_FOLDER / model_name}.pth")

def load_hyperparams(model_name):
    with open(str(HYPERPARAMETERS_FOLDER / model_name) + ".json", 'r') as f:
        return json.load(f)

def load_model(model_class: object, model_name):
    with open(str(HYPERPARAMETERS_FOLDER / model_name) + ".json", 'r') as f:
        hyperparams = json.load(f)

    model = model_class(**hyperparams)
    model.load_state_dict(torch.load(str(MODELS_FOLDER / model_name) + ".pth", weights_only=True, map_location=torch.device('cpu')), strict=True)
    model.eval()
    return model

def plot_metrics(filename):
    """
    Lee un archivo de métricas JSON y genera un gráfico de convergencia 
    por cada métrica registrada, mostrando las curvas de train y val.
    """
    file_path = EXPERIMENTS_FOLDER / filename
    
    with open(file_path, 'r', encoding='utf-8') as f:
        metrics_data = json.load(f)
        
    if not metrics_data:
        print("El archivo de métricas está vacío.")
        return

    # Extraer las métricas base eliminando el prefijo 'train_'
    first_entry = metrics_data[0]
    base_metrics = [
        key.replace("train_", "") 
        for key in first_entry.keys() if key.startswith("train_")
    ]
    
    epochs = [entry["epoch"] for entry in metrics_data]

    # Generar una figura independiente para cada métrica
    for metric in base_metrics:
        train_values = [entry[f"train_{metric}"] for entry in metrics_data]
        val_values = [entry[f"val_{metric}"] for entry in metrics_data]

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_values, label=f'Train {metric}', marker='o')
        plt.plot(epochs, val_values, label=f'Validation {metric}', marker='s')
        
        plt.title(f'Convergencia - {metric}')
        plt.xlabel('Época')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()