import matplotlib.pyplot as plt
import h5py
import numpy as np
from settings import DATA_FOLDER
from data.preprocessing import save_h5_dataset

def rotate_coords(coords: np.ndarray, angle: float) -> np.ndarray:
    """
    Rota las coordenadas (Batch, max_cities, 2) un ángulo específico en grados
    alrededor del centro del mapa normalizado (0.5, 0.5).
    """
    if angle % 360 == 0:
        return coords.copy()

    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    
    rotation_matrix = np.array([
        [c, -s],
        [s,  c]
    ], dtype=np.float32)
    
    # Trasladar las coordenadas al origen (centro en 0.0)
    centered_coords = coords - 0.5
    
    # Aplicar la rotación
    rotated_coords = np.dot(centered_coords, rotation_matrix.T)
    
    # Trasladar de vuelta a la posición original
    final_coords = rotated_coords + 0.5
    
    return final_coords

def augment_train_set(train_filename: str, angles: list = [0, 90, 180, 270]):
    """
    Toma un dataset de Train ya creado y le aplica Data Augmentation mediante rotaciones.
    """
    input_path = DATA_FOLDER / train_filename
    print(f"Cargando Train set base: {input_path}")
    
    with h5py.File(input_path, "r") as f:
        input_keys = list(f['input'].attrs['key_order'])
        output_keys = list(f['output'].attrs['key_order'])
        
        train_inputs_base = {k: f[f'input/{k}'][()] for k in input_keys}
        train_outputs_base = {k: f[f'output/{k}'][()] for k in output_keys}
        
    # --- CAMBIO: Print dinámico basado en los ángulos recibidos ---
    angles_str = ", ".join([f"{a}°" for a in angles])
    print(f"Aplicando rotaciones ({angles_str})...")
    
    aug_inputs = {k: [] for k in input_keys}
    aug_outputs = {k: [] for k in output_keys}
    
    # --- CAMBIO: Iteramos sobre los ángulos parametrizados ---
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
    
    total_train_aug = len(final_train_inputs[input_keys[0]])
    shuffle_idx = np.random.permutation(total_train_aug)
    
    final_train_inputs = {k: v[shuffle_idx] for k, v in final_train_inputs.items()}
    final_train_outputs = {k: v[shuffle_idx] for k, v in final_train_outputs.items()}

    base_name = train_filename.replace("_train.h5", "").replace(".h5", "")
    train_aug_filename = f"{base_name}_train_aug.h5"
    
    print(f"Guardando Train aumentado en: {train_aug_filename}")
    save_h5_dataset(DATA_FOLDER / train_aug_filename, final_train_inputs, final_train_outputs, input_keys, output_keys)
    
    print(f"** ¡Finalizado! El conjunto de entrenamiento creció a {total_train_aug} muestras.")

def plot_rotations(train_filename: str, instance_idx: int, angles: list = [0, 90, 180, 270]):
    """
    Grafica una nube de puntos específica bajo distintas rotaciones para visualizar
    el efecto del Data Augmentation.
    """
    input_path = DATA_FOLDER / train_filename
    
    # 1. Extraer las coordenadas de la instancia elegida
    with h5py.File(input_path, "r") as f:
        # coords tiene forma (Batch, max_cities, 2). Extraemos una sola: (max_cities, 2)
        coords = f['input/coords'][instance_idx] 
        
    # 2. Configurar el lienzo (1 fila, N columnas)
    fig, axes = plt.subplots(1, len(angles), figsize=(4 * len(angles), 4))
    
    # Si solo se pasa un ángulo, convertimos el axe en lista para poder iterar
    if len(angles) == 1:
        axes = [axes]
        
    for ax, angle in zip(axes, angles):
        # 3. Rotar las coordenadas
        # rotate_coords espera formato batch, así que agregamos una dimensión temporalmente
        coords_batch = coords[np.newaxis, ...] 
        rotated = rotate_coords(coords_batch, angle)[0] # Quitamos la dimensión batch
        
        # Separar X e Y para graficar
        x, y = rotated[:, 0], rotated[:, 1]
        
        # 4. Dibujar la nube de puntos
        ax.scatter(x, y, c='cornflowerblue', edgecolors='black', s=50, zorder=2)
        
        # Dibujar el centro de rotación explícitamente
        ax.scatter([0.5], [0.5], c='red', marker='x', s=100, linewidths=2, label='Pivote (0.5, 0.5)', zorder=3)
        
        # 5. Formatear el gráfico para que sea matemáticamente preciso
        ax.set_title(f"Rotación: {angle}°", fontsize=14)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect('equal') # Evita deformaciones visuales
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Poner la leyenda solo en el primer gráfico
        if angle == angles[0]:
            ax.legend(loc='upper right')

    plt.tight_layout()
    plt.show()