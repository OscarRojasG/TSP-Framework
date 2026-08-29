import torch
import torch.nn as nn
from models.base.transformer import Transformer

class TSPCostTransformer(Transformer):
    def __init__(self, input_dim=2, embed_dim=128, num_heads=8, num_encoder_layers=2, dropout_rate=0.1):
        super().__init__(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            dropout_rate=dropout_rate,
        )
        self.embed_dim = embed_dim
        
        # =====================================================
        # --- ENCODER ---
        # =====================================================
        self.encoder_input_layer = nn.Linear(input_dim, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dropout=dropout_rate,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers, enable_nested_tensor=False)
        
        # =====================================================
        # --- COST HEAD ---
        # =====================================================
        self.cost_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )

    def _get_pad_mask(self, max_len, num_cities, device):
        idx = torch.arange(max_len, device=device).unsqueeze(0)
        return idx >= num_cities.unsqueeze(1)

    # =====================================================
    # 1. --- ENCODER (Generación y Predicción) ---
    # =====================================================
    def encode(self, coords, visited, num_cities):
        """
        Procesa el grafo completo y predice el costo directamente.
        Retorna el tensor escalar del costo (Batch,).
        """
        B, max_cities, _ = coords.shape
        device = coords.device

        # 1. Transformer Encoder
        pad_mask = self._get_pad_mask(max_cities, num_cities, device)
        enc_input = self.encoder_input_layer(coords)
        memory = self.encoder(enc_input, src_key_padding_mask=pad_mask) 
        
        # 2. Agregación (Mean Pooling)
        valid_mask = (~pad_mask).unsqueeze(-1).float()  # (B, max_cities, 1)
        sum_memory = (memory * valid_mask).sum(dim=1)   # (B, embed_dim)
        count_nodes = valid_mask.sum(dim=1).clamp(min=1)
        
        graph_embedding = sum_memory / count_nodes      # (B, embed_dim)

        # 3. Predicción del costo
        cost_pred = self.cost_head(graph_embedding)     # (B, 1)

        return cost_pred.squeeze(-1)

    # =====================================================
    # 2. --- DECODER (Pass-through) ---
    # =====================================================
    def decode(self, cost_pred, coords, visited, num_cities):
        """
        La predicción ya se realizó en el encode(). 
        Retornamos el valor directamente para mantener la compatibilidad de firmas.
        """
        return cost_pred