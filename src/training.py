from torch.utils.data import DataLoader, TensorDataset, random_split
import torch
from torch import nn
import os
import numpy as np
from .data_gen import read_train_data
from .settings import models_path as model_folder_path
import copy


class Metrics:
    def __init__(self):
        self.loss_history = []
        self.acc_history = []
        self.reset_epoch()

    def reset_epoch(self):
        self.loss_sum = 0
        self.correct = 0
        self.total = 0

    def update_batch(self, outputs, targets, loss, batch_size):
        self.loss_sum += loss * batch_size
        self.total += batch_size
        _, predicted = torch.max(outputs.data, 1)
        self.correct += (predicted == targets.argmax(dim=1)).sum().item()

    def end_epoch(self):
        epoch_loss = self.loss_sum / self.total if self.total > 0 else 0
        epoch_acc = 100 * self.correct / self.total if self.total > 0 else 0
        self.loss_history.append(epoch_loss)
        self.acc_history.append(epoch_acc)
        self.reset_epoch()


def load_data(file_path):
    X_src, visited, Y = read_train_data(file_path)
    X_src = torch.tensor(np.array(X_src), dtype=torch.float32)
    visited = torch.tensor(np.array(visited), dtype=torch.int32)
    Y = torch.tensor(np.array(Y), dtype=torch.int32)
    dataset = TensorDataset(X_src, visited, Y)
    return dataset

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
        # --- ENTRENAMIENTO ---
        model.train()
        for X_src_batch, visited_batch, y_batch in train_loader:
            # Mover los datos al dispositivo
            X_src_batch = X_src_batch.to(device)
            visited_batch = visited_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model.forward(X_src_batch, visited_batch)
            loss = loss_function(outputs, y_batch.argmax(dim=-1))
            loss.backward()
            optimizer.step()

            train_metrics.update_batch(outputs, y_batch, loss.item(), X_src_batch.size(0))
        train_metrics.end_epoch()

        # --- VALIDACIÓN ---
        model.eval()
        with torch.no_grad():
            for X_src_batch, visited_batch, y_batch in test_loader:
                # Mover los datos al dispositivo
                X_src_batch = X_src_batch.to(device)
                visited_batch = visited_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model.forward(X_src_batch, visited_batch)
                loss = loss_function(outputs, y_batch.argmax(dim=-1))

                test_metrics.update_batch(outputs, y_batch, loss.item(), X_src_batch.size(0))
        test_metrics.end_epoch()

        print(f'Epoch {epoch + 1}/{epochs} - '
            f'Train Loss: {train_metrics.loss_history[-1]:.4f}, '
            f'Train Accuracy: {train_metrics.acc_history[-1]:.2f}% - '
            f'Val Loss: {test_metrics.loss_history[-1]:.4f}, Val Accuracy: {test_metrics.acc_history[-1]:.2f}%')
        
        # Actualizar mejor modelo
        if test_metrics.acc_history[-1] > best_acc:
            best_acc = test_metrics.acc_history[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

    # Cargar los mejores pesos del modelo
    model.load_state_dict(best_model_wts)
    return model
        
def save_model(model, filename):
    os.makedirs(model_folder_path, exist_ok=True)
    torch.save(model.state_dict(), model_folder_path + filename)

def load_model(model: nn.Module, filename):
    model.load_state_dict(torch.load(model_folder_path + filename, weights_only=True), strict=True)
    model.eval()
    return model