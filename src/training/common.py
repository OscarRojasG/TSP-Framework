import os
import torch
import json
from dataclasses import dataclass
from settings import MODELS_FOLDER, HYPERPARAMETERS_FOLDER

@dataclass
class LRConfig:
    value: float            # Tasa de aprendizaje inicial
    factor: float = 0.5     # Factor de reducción
    patience: int = 999999  # Épocas sin mejora antes de reducir el LR
    min: float = 0.0        # Tasa de aprendizaje mínima permitida

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