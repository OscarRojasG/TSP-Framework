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