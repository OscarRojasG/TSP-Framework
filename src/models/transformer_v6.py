import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base.transformer import Transformer


class TSPTransformer(Transformer):
    def __init__(
        self,
        input_dim=2,
        embed_dim=128,
        num_heads=8,
        num_encoder_layers=2,
        num_glimpses=2,
        dropout_rate=0.1,
        dist_bias_scale=1.0,   # NUEVO: peso del bias por distancia
    ):
        super().__init__(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_glimpses=num_glimpses,
            dropout_rate=dropout_rate,
        )
        self.embed_dim = embed_dim
        self.dist_bias_scale = dist_bias_scale

        # --- ENCODER ---
        self.encoder_input_layer = nn.Linear(input_dim, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # --- DECODER ---
        self.ctx_fusion = nn.Linear(3 * embed_dim, embed_dim)

        self.num_glimpses = num_glimpses
        self.glimpse_proj = nn.Linear(embed_dim, embed_dim)

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
    # =====================================================
    def encode(self, x_src):
        """
        x_src: (batch, num_cities, 2)
        """
        enc_input = self.encoder_input_layer(x_src)
        memory = self.encoder(enc_input)
        return memory

    # =====================================================
    # 2. --- DECODER ---
    # =====================================================
    def decode(self, memory, coords, visited):
        """
        memory:  (batch, num_cities, embed_dim)
        coords:  (batch, num_cities, 2)
        visited: (batch, T)
        """
        B, num_cities, _ = memory.shape
        device = memory.device

        # --- MÁSCARA CIUDADES VISITADAS ---
        visited_mask_pos = visited != -1

        visited_city_mask = torch.zeros(
            B, num_cities, dtype=torch.bool, device=device
        )

        valid = visited != -1
        batch_ids, pos_ids = valid.nonzero(as_tuple=True)
        visited_city_mask[batch_ids, visited[batch_ids, pos_ids]] = True

        # --- CONTEXTO: media visitadas ---
        mask = visited_city_mask.unsqueeze(-1)
        sum_ctx = (memory * mask).sum(dim=1)
        count_ctx = mask.sum(dim=1).clamp(min=1)
        context_mean = sum_ctx / count_ctx

        # --- PRIMERA Y ÚLTIMA CIUDAD ---
        start_idx = visited_mask_pos.float().argmax(dim=1)
        last_idx = visited_mask_pos.sum(dim=1) - 1

        batch_idx = torch.arange(B, device=device)

        start_city_embed = memory[
            batch_idx, visited[batch_idx, start_idx].long()
        ]
        last_city_embed = memory[
            batch_idx, visited[batch_idx, last_idx].long()
        ]

        # --- FUSIÓN DE CONTEXTO ---
        ctx_concat = torch.cat(
            [context_mean, last_city_embed, start_city_embed], dim=-1
        )
        decoder_state = self.ctx_fusion(ctx_concat)

        # --- GLIMPSE (CROSS-ATTENTION) ---
        query = self.glimpse_proj(decoder_state).unsqueeze(1)

        for _ in range(self.num_glimpses):
            attn_out, _ = self.cross_attn(
                query=query,
                key=memory,
                value=memory,
                key_padding_mask=visited_city_mask
            )
            query = self.norm1(attn_out + query)
            ff_out = self.ff(query)
            query = self.norm2(ff_out + query)

        attn_out = query.squeeze(1)

        # --- POINTER SCORING ---
        ptr_query = self.pointer_proj(attn_out)

        scores = torch.matmul(
            ptr_query.unsqueeze(1),
            memory.transpose(1, 2)
        ).squeeze(1)

        scores = scores / math.sqrt(self.embed_dim)

        # =====================================================
        # NUEVO: BIAS POR DISTANCIA
        # =====================================================
        last_city_coords = coords[
            batch_idx, visited[batch_idx, last_idx].long()
        ]  # (B, 2)

        distances = torch.norm(
            coords - last_city_coords.unsqueeze(1),
            dim=-1
        )  # (B, N)

        distance_bias = -distances
        scores = scores + self.dist_bias_scale * distance_bias

        # --- MASKING Y SOFTMAX ---
        scores = scores.masked_fill(visited_city_mask, float("-inf"))
        probs = F.softmax(scores, dim=-1)

        return probs
    
    def forward(self, x_src, visited):
        memory = self.encode(x_src)
        return self.decode(memory, x_src, visited)