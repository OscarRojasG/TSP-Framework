import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TSPTransformer(nn.Module):
    def __init__(self, input_dim=2, embed_dim=128, num_heads=8, num_layers=2, dropout_rate=0.1):
        super(TSPTransformer, self).__init__()
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
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # --- DECODER ---
        # Fusión contexto (media de visitadas + primera + última)
        self.ctx_fusion = nn.Linear(3 * embed_dim, embed_dim)

        # Cross Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )

        # Pointer Scorer
        self.pointer = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x_src, visited):
        """
        x_src: (batch, num_cities, 2) -> Coordenadas
        visited: (batch, num_cities) -> Índice ciudades visitadas (-1 para padding)
        """
        B, num_cities, _ = x_src.shape

        # 1. --- ENCODER ---
        # Proyectamos las coordenadas y pasamos por el encoder
        enc_input = self.encoder_input_layer(x_src)
        memory = self.encoder(enc_input) # (batch, n_cities, embed_dim)

        # 2. --- MÁSCARA CIUDADES VISITADAS ---
        # Posiciones válidas del tour
        visited_mask_pos = visited != -1          # (B, T)

        # Máscara por ciudad
        visited_city_mask = torch.zeros(B, num_cities, dtype=torch.bool, device=visited.device)

        valid = visited != -1
        batch_ids, pos_ids = valid.nonzero(as_tuple=True)

        visited_city_mask[batch_ids, visited[batch_ids, pos_ids]] = True

        # 3. --- DECODER: Media de ciudades visitadas ---
        mask = visited_city_mask.unsqueeze(-1)    # (B, N, 1)

        sum_ctx = (memory * mask).sum(dim=1)
        count_ctx = mask.sum(dim=1)
        context_mean = sum_ctx / count_ctx         # (B, D)

        # 4. --- DECODER: Obtener primera y última ciudad y concatenar a la media ---
        start_idx = visited_mask_pos.float().argmax(dim=1)
        last_idx = visited_mask_pos.sum(dim=1) - 1

        batch_idx = torch.arange(B, device=x_src.device)
        start_city_embed = memory[batch_idx, visited[batch_idx, start_idx].long()]  # (B, D)
        last_city_embed = memory[batch_idx, visited[batch_idx, last_idx].long()]    # (B, D)

        # Concatenación y proyección
        ctx_concat = torch.cat([context_mean, last_city_embed, start_city_embed], dim=-1)  # (B, 3D)
        decoder_query = self.ctx_fusion(ctx_concat)  # (B, D)

        # 5. --- DECODER: Cross-Attention (Glimpse) ---
        query = decoder_query.unsqueeze(1)  # (B, 1, D)

        attn_out, _ = self.cross_attn(
            query=query,        # (B, 1, D)
            key=memory,         # (B, N, D)
            value=memory,      # (B, N, D)
        )

        attn_out = attn_out.squeeze(1)        # (B, D)

        # 5. --- DECODER: Pointer scoring ---
        ptr_query = self.pointer(attn_out)             # (B, D)

        scores = torch.matmul(
            ptr_query.unsqueeze(1),                    # (B, 1, D)
            memory.transpose(1, 2)                     # (B, D, N)
        ).squeeze(1)                                   # (B, N)

        # 6. --- DECODER: Masking y Softmax ---
        scores = scores.masked_fill(visited_city_mask, float("-inf"))
        probs = F.softmax(scores, dim=-1)              # (B, N)

        return probs