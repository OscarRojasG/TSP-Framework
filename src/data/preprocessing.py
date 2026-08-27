import torch
import h5py
from torch.utils.data import Dataset
import os
import numpy as np
from settings import DATA_FOLDER

class H5Dataset(Dataset):
    def __init__(self, filepath, max_size=None):
        # Permite pasar la ruta completa o solo el nombre del archivo
        self.filepath = filepath if isinstance(filepath, str) and filepath.startswith(str(DATA_FOLDER)) else DATA_FOLDER / filepath
        self.name = os.path.basename(self.filepath)
        self.file = None

        with h5py.File(self.filepath, "r") as f:
            self.input_keys = list(f['input'].attrs['key_order'])
            self.output_keys = list(f['output'].attrs['key_order'])
            
            total_len = len(f['input'][self.input_keys[0]])
            self.dataset_len = total_len if max_size is None else min(total_len, max_size)

    def _open_file(self):
        self.file = h5py.File(self.filepath, "r")
        self.input_datasets = {k: self.file[f'input/{k}'] for k in self.input_keys}
        self.output_datasets = {k: self.file[f'output/{k}'] for k in self.output_keys}
        
    def _to_tensor(self, val):
        if isinstance(val, np.ndarray):
            return torch.from_numpy(val)
        return torch.tensor(val)

    def __getitem__(self, idx):
        if self.file is None: 
            self._open_file()
            
        inputs = [self._to_tensor(self.input_datasets[k][idx]) for k in self.input_keys]
        outputs = [self._to_tensor(self.output_datasets[k][idx]) for k in self.output_keys]
        return tuple(inputs), tuple(outputs)
    
    def __len__(self):
        return self.dataset_len

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state['file'] = None
        state['input_datasets'] = None
        state['output_datasets'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.file = None

def load_dataset(filepath, max_size=None, verbose=True):
    dataset = H5Dataset(filepath, max_size)
    if verbose:
        print(f"Dataset {dataset.name} cargado con {len(dataset)} muestras.")
    return dataset

def split_dataset(filename: str, train_size: int, seed: int = 42) -> tuple:
    """
    Lee el dataset original, lo mezcla y lo divide en Train y Test.
    Retorna los nombres de los nuevos archivos generados.
    """
    input_path = DATA_FOLDER / filename
    print(f"Cargando dataset original: {input_path}")
    
    with h5py.File(input_path, "r") as f:
        input_keys = list(f['input'].attrs['key_order'])
        output_keys = list(f['output'].attrs['key_order'])
        
        inputs = {k: f[f'input/{k}'][()] for k in input_keys}
        outputs = {k: f[f'output/{k}'][()] for k in output_keys}
        
    total_samples = len(inputs[input_keys[0]])
    
    np.random.seed(seed)
    indices = np.random.permutation(total_samples)
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    # --- TRAIN SET (Base) ---
    train_inputs = {k: inputs[k][train_indices] for k in input_keys}
    train_outputs = {k: outputs[k][train_indices] for k in output_keys}

    # --- TEST SET (Intacto) ---
    test_inputs = {k: inputs[k][test_indices] for k in input_keys}
    test_outputs = {k: outputs[k][test_indices] for k in output_keys}

    # Crear nombres de archivo
    base_name = filename.replace(".h5", "").replace(".data", "")
    train_filename = f"{base_name}_train.h5"
    test_filename = f"{base_name}_test.h5"
    
    print(f"Guardando Train puro ({len(train_indices)} muestras) en: {train_filename}")
    save_h5_dataset(DATA_FOLDER / train_filename, train_inputs, train_outputs, input_keys, output_keys)
    
    print(f"Guardando Test puro ({len(test_indices)} muestras) en: {test_filename}")
    save_h5_dataset(DATA_FOLDER / test_filename, test_inputs, test_outputs, input_keys, output_keys)
    
    return train_filename, test_filename

def save_h5_dataset(filepath, input_data, output_data, input_keys, output_keys):
    """
    Guarda los diccionarios de datos en formato HDF5.
    """
    with h5py.File(filepath, "w") as f:
        g_input = f.create_group("input")
        g_output = f.create_group("output")

        for key in input_keys:
            g_input.create_dataset(key, data=input_data[key])
        g_input.attrs['key_order'] = input_keys

        for key in output_keys:
            g_output.create_dataset(key, data=output_data[key])
        g_output.attrs['key_order'] = output_keys