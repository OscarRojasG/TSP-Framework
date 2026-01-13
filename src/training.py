from torch.utils.data import DataLoader, TensorDataset, random_split
import torch
from torch import nn
import os
import numpy as np
from .data_gen import read_train_data
from .settings import models_path, hyperparams_path
import copy
import json


class Metrics:
    def __init__(self):
        self.loss_history = []
        self.acc_history = []

    def add_epoch(self, loss, acc):
        self.loss_history.append(loss)
        self.acc_history.append(acc)


def load_data(file_path):
    X_src, visited, Y = read_train_data(file_path)
    X_src = torch.tensor(np.array(X_src), dtype=torch.float32)
    visited = torch.tensor(np.array(visited), dtype=torch.int32)
    Y = torch.tensor(np.array(Y), dtype=torch.int32)
    dataset = TensorDataset(X_src, visited, Y)
    return dataset

def train_epoch(model: nn.Module, train_loader: DataLoader, loss_function, optimizer, device):
    model.train()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for X_src_batch, visited_batch, y_batch in train_loader:
        X_src_batch = X_src_batch.to(device)
        visited_batch = visited_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        outputs = model(X_src_batch, visited_batch)

        labels = y_batch.argmax(dim=-1)
        loss = loss_function(outputs, labels)

        loss.backward()
        optimizer.step()

        # Acumulación de métricas
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    loss = total_loss / total_samples
    accuracy = 100 * total_correct / total_samples

    return loss, accuracy

def val_epoch(model: nn.Module, val_loader: DataLoader, loss_function, device):
    model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for X_src_batch, visited_batch, y_batch in val_loader:
            X_src_batch = X_src_batch.to(device)
            visited_batch = visited_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_src_batch, visited_batch)

            labels = y_batch.argmax(dim=-1)
            loss = loss_function(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size

    loss = total_loss / total_samples
    accuracy = 100 * total_correct / total_samples

    return loss, accuracy

def train(model: nn.Module, dataset, epochs, train_size, test_size, batch_size, learning_rate, seed=42):
    # --- CONFIGURAR DISPOSITIVO ---
    device = torch.device("cuda" if torch.cuda.is_available() 
                          else "mps" if torch.backends.mps.is_available() 
                          else "cpu")
    print(f"Usando dispositivo: {device}")

    torch.manual_seed(seed)
    torch.set_num_threads(os.cpu_count())

    # Mover el modelo al dispositivo
    model = model.to(device)

    # Generar dataset de validación y entrenamiento
    test_set, complement_set = random_split(dataset, [test_size, len(dataset) - test_size])
    train_set, _ = random_split(complement_set, [train_size, len(complement_set) - train_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)

    # Función de pérdida y optimizador
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Métricas
    train_metrics = Metrics()
    test_metrics = Metrics()

    # Guardar mejor modelo
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = float("-inf")

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, loss_function, optimizer, device)
        val_loss, val_acc = val_epoch(model, test_loader, loss_function, device)

        train_metrics.add_epoch(train_loss, train_acc)
        test_metrics.add_epoch(val_loss, val_acc)

        print(f'Epoch {epoch + 1}/{epochs} - '
            f'Train Loss: {train_loss:.4f}, '
            f'Train Accuracy: {train_acc:.2f}% - '
            f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%')

        # Actualizar mejor modelo
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    # Cargar los mejores pesos del modelo
    model.load_state_dict(best_model_wts)
    return model
        
def save_model(model, model_name):
    os.makedirs(hyperparams_path, exist_ok=True)
    with open(hyperparams_path + model_name + ".json", 'w') as f:
        json.dump(model.hyperparams, f, indent=4)

    os.makedirs(models_path, exist_ok=True)
    torch.save(model.state_dict(), models_path + model_name + ".pth")

def load_hyperparams(model_name):
    with open(hyperparams_path + model_name + ".json", 'r') as f:
        return json.load(f)

def load_model(model_class: object, model_name):
    with open(hyperparams_path + model_name + ".json", 'r') as f:
        hyperparams = json.load(f)

    model = model_class(**hyperparams)
    model.load_state_dict(torch.load(models_path + model_name + ".pth", weights_only=True), strict=True)
    model.eval()
    return model