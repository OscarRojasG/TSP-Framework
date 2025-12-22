import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base.transformer import Transformer

class TSPTransformer(Transformer):
    def __init__(self, input_dim=2, embed_dim=128, num_heads=8, num_encoder_layers=2, num_glimpses=2, dropout_rate=0.1):
        super().__init__(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_glimpses=num_glimpses,
            dropout_rate=dropout_rate,
        )
        self.embed_dim = embed_dim
        
        # --- ENCODER ---
        # Proyección inicial de coordenadas (x, y)
        self.encoder_input_layer = nn.Linear(input_dim, embed_dim)
        
        # Transformer Encoder (Sin Positional Encoding)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dropout=dropout_rate,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # --- DECODER ---
        # Fusión contexto (media de visitadas + primera + última)
        self.ctx_fusion = nn.Linear(3 * embed_dim, embed_dim)

        # Cross Attention
        self.num_glimpses = num_glimpses
        self.glimpse_proj = nn.Linear(embed_dim, embed_dim) # NUEVO: Proyección antes de glimpse

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        # Pointer Scorer
        self.pointer_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    # =====================================================
    # 1. --- ENCODER ---
    # Se llama una sola vez por instancia
    # =====================================================
    def encode(self, x_src):
        """
        x_src: (batch, num_cities, 2) -> Coordenadas
        """
        # Proyectamos las coordenadas y pasamos por el encoder
        enc_input = self.encoder_input_layer(x_src)
        memory = self.encoder(enc_input)  # (batch, n_cities, embed_dim)

        return memory

    # =====================================================
    # 2. --- DECODER ---
    # Se llama en cada paso del rollout
    # =====================================================
    def decode(self, memory, visited):
        """
        memory:  (batch, num_cities, embed_dim) -> Salida del encoder
        visited: (batch, T) -> Índice ciudades visitadas (-1 para padding)
        """
        B, num_cities, _ = memory.shape
        device = memory.device

        # 2. --- MÁSCARA CIUDADES VISITADAS ---
        # Posiciones válidas del tour
        visited_mask_pos = visited != -1          # (B, T)

        # Máscara por ciudad
        visited_city_mask = torch.zeros(
            B, num_cities, dtype=torch.bool, device=device
        )

        valid = visited != -1
        batch_ids, pos_ids = valid.nonzero(as_tuple=True)

        visited_city_mask[batch_ids, visited[batch_ids, pos_ids]] = True

        # 3. --- DECODER: Media de ciudades visitadas ---
        mask = visited_city_mask.unsqueeze(-1)    # (B, N, 1)

        sum_ctx = (memory * mask).sum(dim=1)
        count_ctx = mask.sum(dim=1).clamp(min=1)
        context_mean = sum_ctx / count_ctx         # (B, D)

        # 4. --- DECODER: Obtener primera y última ciudad y concatenar a la media ---
        start_idx = visited_mask_pos.float().argmax(dim=1)
        last_idx = visited_mask_pos.sum(dim=1) - 1

        batch_idx = torch.arange(B, device=device)
        start_city_embed = memory[
            batch_idx, visited[batch_idx, start_idx].long()
        ]  # (B, D)
        last_city_embed = memory[
            batch_idx, visited[batch_idx, last_idx].long()
        ]   # (B, D)

        # Concatenación y proyección
        ctx_concat = torch.cat(
            [context_mean, last_city_embed, start_city_embed], dim=-1
        )  # (B, 3D)
        decoder_state = self.ctx_fusion(ctx_concat)  # (B, D)

        # 5. --- DECODER: Cross-Attention (Glimpse) ---
        query = self.glimpse_proj(decoder_state).unsqueeze(1)

        for _ in range(self.num_glimpses):
            attn_out, _ = self.cross_attn(
                query=query,            # (B, 1, D)
                key=memory,             # (B, N, D)
                value=memory,           # (B, N, D)
                key_padding_mask=visited_city_mask  # (B, N)
            )

            query = self.norm1(attn_out + query)   # Residual + Norm
            ff_out = self.ff(query)                # Feed-Forward
            query = self.norm2(ff_out + query)  # Residual + Norm

        attn_out = query.squeeze(1)         # (B, D)

        # 6. --- DECODER: Pointer scoring ---
        ptr_query = self.pointer_proj(attn_out)        # (B, D)

        scores = torch.matmul(
            ptr_query.unsqueeze(1),                    # (B, 1, D)
            memory.transpose(1, 2)                     # (B, D, N)
        ).squeeze(1)                                   # (B, N)

        scores = scores / math.sqrt(self.embed_dim)    # Escalado de scores

        # 7. --- DECODER: Masking y Softmax ---
        scores = scores.masked_fill(visited_city_mask, float("-inf"))
        probs = F.softmax(scores, dim=-1)              # (B, N)

        return probs