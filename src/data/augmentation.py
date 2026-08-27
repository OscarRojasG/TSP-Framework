import h5py
import numpy as np
from settings import DATA_FOLDER
from data.preprocessing import save_h5_dataset

def rotate_coords(coords: np.ndarray, angle: float) -> np.ndarray:
    """
    Rota las coordenadas (Batch, max_cities, 2) un ángulo específico en grados.
    """
    if angle == 0:
        return coords.copy()

    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    
    rotation_matrix = np.array([
        [c, -s],
        [s,  c]
    ], dtype=np.float32)
    
    return np.dot(coords, rotation_matrix.T)

def augment_train_set(train_filename: str):
    """
    Toma un dataset de Train ya creado y le aplica Data Augmentation.
    """
    input_path = DATA_FOLDER / train_filename
    print(f"Cargando Train set base: {input_path}")
    
    with h5py.File(input_path, "r") as f:
        input_keys = list(f['input'].attrs['key_order'])
        output_keys = list(f['output'].attrs['key_order'])
        
        train_inputs_base = {k: f[f'input/{k}'][()] for k in input_keys}
        train_outputs_base = {k: f[f'output/{k}'][()] for k in output_keys}
        
    print("Aplicando rotaciones (0°, 90°, 180°, 270°)...")
    aug_inputs = {k: [] for k in input_keys}
    aug_outputs = {k: [] for k in output_keys}
    
    angles = [0, 90, 180, 270]
    
    for angle in angles:
        for k in output_keys:
            aug_outputs[k].append(train_outputs_base[k].copy())
            
        for k in input_keys:
            if k == "coords":
                rotated_coords = rotate_coords(train_inputs_base[k], angle)
                aug_inputs[k].append(rotated_coords)
            else:
                aug_inputs[k].append(train_inputs_base[k].copy())
                
    print("Ensamblando y mezclando el Train set final...")
    final_train_inputs = {k: np.concatenate(aug_inputs[k], axis=0) for k in input_keys}
    final_train_outputs = {k: np.concatenate(aug_outputs[k], axis=0) for k in output_keys}
    
    # Mezclamos para distribuir las rotaciones uniformemente
    total_train_aug = len(final_train_inputs[input_keys[0]])
    shuffle_idx = np.random.permutation(total_train_aug)
    
    final_train_inputs = {k: v[shuffle_idx] for k, v in final_train_inputs.items()}
    final_train_outputs = {k: v[shuffle_idx] for k, v in final_train_outputs.items()}

    base_name = train_filename.replace("_train.h5", "").replace(".h5", "")
    train_aug_filename = f"{base_name}_train_aug.h5"
    
    print(f"Guardando Train aumentado en: {train_aug_filename}")
    save_h5_dataset(DATA_FOLDER / train_aug_filename, final_train_inputs, final_train_outputs, input_keys, output_keys)
    
    print(f"** ¡Finalizado! El conjunto de entrenamiento creció a {total_train_aug} muestras.")