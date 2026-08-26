import torch
import numpy as np
from TSP import TSPState
from models.base.transformer import Transformer
from solvers.solver import Solver

class ModelSolver(Solver):
    def __init__(self, model: Transformer, input_adapter):
        super().__init__("ModelSolver")
        self.model = model
        self.input_adapter = input_adapter
        
        # Inferimos el dispositivo (CPU/GPU) donde está alojado el modelo
        self.device = next(model.parameters()).device 

    def _get_tensor_inputs(self, state: TSPState):
        """
        Usa el input_adapter para extraer las características del estado
        y las convierte en tensores listos para el modelo (con dimensión de batch).
        """
        vecs = self.input_adapter.input_2_vec(state)
        
        tensor_inputs = []
        for v in vecs:
            if isinstance(v, np.ndarray):
                tensor = torch.from_numpy(v)
            else:
                tensor = torch.tensor(v)
                
            # Añadimos unsqueeze(0) para simular un Batch de tamaño 1
            tensor_inputs.append(tensor.unsqueeze(0).to(self.device))
            
        return tensor_inputs

    @torch.no_grad()
    def solve_instance(self, instance):
        """
        Resuelve una sola instancia simulando el paso a paso del decoder.
        """
        state = TSPState(instance)

        # 1. Obtener inputs iniciales y generar la memoria latente (Encoder)
        inputs = self._get_tensor_inputs(state)
        memory = self.model.encode(*inputs)

        # 2. Rollout autoregresivo (Decoder)
        while not state.is_finished():
            current_inputs = self._get_tensor_inputs(state)

            # Obtener logits de la próxima ciudad
            logits = self.model.decode(memory, *current_inputs)
            
            # Seleccionamos el índice de la ciudad con mayor probabilidad/score
            next_city = logits.argmax(dim=-1).item()
            
            # Actualizamos el estado interno
            state.visit_city(next_city)

        return state