import optuna
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
from .training import train_epoch, val_epoch
import random, numpy as np

optuna.logging.set_verbosity(optuna.logging.WARNING)

class OptunaCategorical:
    def __init__(self, name, choices):
        self.name = name
        self.choices = choices

    def suggest(self, trial):
        return trial.suggest_categorical(self.name, self.choices)

class OptunaContinuous:
    def __init__(self, name, low, high):
        self.name = name
        self.low = low
        self.high = high

    def suggest(self, trial):
        return trial.suggest_float(self.name, self.low, self.high)

class OptunaLR(OptunaContinuous):
    def __init__(self, low, high):
        super().__init__("lr", low, high)

class OptunaBatchSize(OptunaCategorical):
    def __init__(self, choices):
        super().__init__("batch_size", choices)

class OptunaHyperparams:
    def __init__(self, lr: OptunaLR, batch_size: OptunaBatchSize):
        self.lr = lr
        self.batch_size = batch_size
        self.model_hyperparams = []

    def register(self, hyperparam: OptunaCategorical | OptunaContinuous):
        self.model_hyperparams.append(hyperparam)

def objective(trial, model_class: object, hyperparams: OptunaHyperparams, epochs: int, train_set, test_set, seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(seed)
    random.seed(seed)

    print(f"🚀 Trial {trial.number} iniciado")

    # Sugerir hiperparámetros
    lr = hyperparams.lr.suggest(trial)
    batch_size = hyperparams.batch_size.suggest(trial)
    model_params = {}
    for param in hyperparams.model_hyperparams:
        model_params[param.name] = param.suggest(trial)

    # Imprimir hiperparámetros
    all_params = {
        "lr": lr,
        "batch_size": batch_size,
        **model_params
    }
    print("📌 Hiperparámetros del trial:")
    for k, v in all_params.items():
        print(f"   - {k}: {v}")

    # Crear modelo
    model = model_class(**model_params).to(device)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Función de pérdida y optimizador
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_loss = float("inf")

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, loss_function, optimizer, device)
        val_loss, val_acc = val_epoch(model, test_loader, loss_function, device)

        print(f'Epoch {epoch + 1}/{epochs} - '
            f'Train Loss: {train_loss:.4f}, '
            f'Train Accuracy: {train_acc:.2f}% - '
            f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%')

        # Actualizar mejor loss
        if val_loss < best_loss:
            best_loss = val_loss

        # Reportamos la LOSS
        trial.report(val_loss, step=epoch)
        
    print(f"✅ Trial {trial.number} terminado | val_loss={val_loss:.4f}")
    return best_loss

def optimize_hyperparams(model_class: object, hyperparams: OptunaHyperparams, n_trials: int, dataset: TensorDataset, epochs: int, train_size: int, test_size: int, seed=42):
    # Generar dataset de validación y entrenamiento
    generator = torch.Generator().manual_seed(seed)
    test_set, complement_set = random_split(dataset, [test_size, len(dataset) - test_size], generator=generator)
    train_set, _ = random_split(complement_set, [train_size, len(complement_set) - train_size], generator=generator)

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, model_class, hyperparams, epochs, train_set, test_set, seed), n_trials=n_trials)

    return study