import json
import torch
import os
import copy
from torch.utils.data import DataLoader, random_split
from training.metrics import EpochMetrics
from training.common import *
from settings import EXPERIMENTS_FOLDER

def train_epoch(model, train_loader, optimizer, loss_fn, metrics, device):
    model.train()

    for inputs_batch, y_batch in train_loader:
        inputs = [i.to(device, non_blocking=True) for i in inputs_batch]
        target = y_batch[0].to(device, non_blocking=True) # Tomamos la salida Y

        optimizer.zero_grad(set_to_none=True)

        logits = model(*inputs)

        loss = loss_fn.step(logits, target)
        for metric in metrics:
            metric.step(logits, target)

        loss.backward()
        optimizer.step()

    loss_val = loss_fn.compute()
    m_values = [m.compute() for m in metrics]
    
    return loss_val, m_values

def val_epoch(model, val_loader, loss_fn, metrics, device):
    model.eval()

    with torch.no_grad():
        for inputs_batch, y_batch in val_loader:
            inputs = [i.to(device, non_blocking=True) for i in inputs_batch]
            target = y_batch[0].to(device, non_blocking=True)

            logits = model(*inputs)

            loss_fn.step(logits, target)
            for metric in metrics:
                metric.step(logits, target)

    loss_val = loss_fn.compute()
    m_values = [m.compute() for m in metrics]
    
    return loss_val, m_values

def generate_sets(dataset, train_size, val_size, seed):
    generator = torch.Generator().manual_seed(seed)
    remaining_size = len(dataset) - train_size - val_size

    train_set, val_set, _ = random_split(
        dataset, 
        [train_size, val_size, remaining_size],
        generator=generator
    )
    return train_set, val_set

def print_epoch_results(loss_fn, train_metrics, val_metrics, metrics):
    train_loss = train_metrics.get_last_value(loss_fn.__class__)
    val_loss = val_metrics.get_last_value(loss_fn.__class__)
    
    print(f"    Train {loss_fn.name}: {loss_fn.format(train_loss)} | Val {loss_fn.name}: {loss_fn.format(val_loss)}")

    # Imprimir el resto de métricas de validación
    metrics_strs = []
    for m in metrics:
        val = val_metrics.get_last_value(m.__class__)
        metrics_strs.append(f"{m.name}: {m.format(val)}")
    
    if metrics_strs:
        print(f"    {' | '.join(metrics_strs)}")

def train(model, epochs, train_set, val_set, batch_size, lr_config: LRConfig, loss_fn, metrics, metrics_filename, patience, weight_decay, device):
    num_workers = os.cpu_count() or 1
    use_pin_memory = device.type in ['cuda', 'mps']

    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, pin_memory=use_pin_memory, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers, pin_memory=use_pin_memory)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_config.value, weight_decay=weight_decay)
    
    # --- CONFIGURACIÓN DEL SCHEDULER ---
    scheduler_mode = 'max' if loss_fn.maximize else 'min'
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=scheduler_mode,
        factor=lr_config.factor,
        patience=lr_config.patience,
        min_lr=lr_config.min
    )

    train_metrics, val_metrics = EpochMetrics(), EpochMetrics()
    history = []  # Lista para ir recolectando las métricas en cada época
    
    best_val_score = float('-inf') if loss_fn.maximize else float('inf')
    best_weights = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        # --- TRAIN ---
        train_loss_val, train_m_vals = train_epoch(model, train_loader, optimizer, loss_fn, metrics, device)
        
        train_metrics.add_value(loss_fn.__class__, train_loss_val)
        for m, val in zip(metrics, train_m_vals):
            train_metrics.add_value(m.__class__, val)

        # --- VAL ---
        val_loss_val, val_m_vals = val_epoch(model, val_loader, loss_fn, metrics, device)
        
        val_metrics.add_value(loss_fn.__class__, val_loss_val)
        for m, val in zip(metrics, val_m_vals):
            val_metrics.add_value(m.__class__, val)

        # --- REGISTRO PARA EL JSON ---
        epoch_data = {
            "epoch": epoch,
            f"train_{loss_fn.name}": float(train_loss_val),
            f"val_{loss_fn.name}": float(val_loss_val)
        }
        for m, t_val, v_val in zip(metrics, train_m_vals, val_m_vals):
            epoch_data[f"train_{m.name}"] = float(t_val)
            epoch_data[f"val_{m.name}"] = float(v_val)
        
        history.append(epoch_data)

        # Imprimir resultados
        print(f"{'\n' if epoch == 1 else ''}Epoch {epoch}/{epochs}")
        print_epoch_results(loss_fn, train_metrics, val_metrics, metrics)

        # --- ACTUALIZAR SCHEDULER ---
        scheduler.step(val_loss_val)

        # Evaluar si es el mejor modelo (considerando si la métrica se maximiza o minimiza)
        is_best = (val_loss_val > best_val_score) if loss_fn.maximize else (val_loss_val < best_val_score)
        
        if is_best:
            best_val_score = val_loss_val
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Early Stopping
        if epochs_without_improvement > patience:
            print(f"** Early stopping en época {epoch}. Sin mejora de métrica objetivo durante {patience} épocas.")
            break

    # --- GUARDAR HISTORIAL EN JSON ---
    os.makedirs(EXPERIMENTS_FOLDER, exist_ok=True)
    file_path = os.path.join(str(EXPERIMENTS_FOLDER), metrics_filename)
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)
    print(f"\n** Historial de entrenamiento guardado en: {file_path}")

    # Restaurar los mejores pesos
    if best_weights is not None:
        model.load_state_dict(best_weights)
        print(f"** Mejor modelo restaurado (Época {best_epoch}): {loss_fn.name} = {loss_fn.format(best_val_score)}")

    return model

def sl_train(model, epochs, train_set, val_set, batch_size, lr_config: LRConfig, loss_fn, metrics, metrics_filename, patience=999999, weight_decay=0, seed=42):
    model, device = config_training(model, seed)
    return train(model, epochs, train_set, val_set, batch_size, lr_config, loss_fn, metrics, metrics_filename, patience, weight_decay, device)