import torch
import torch.nn as nn
import math
from models.base.transformer import Transformer

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
        self.encoder_input_layer = nn.Linear(input_dim, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dropout=dropout_rate,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers, enable_nested_tensor=False)
        
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

        self.pointer_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def _get_pad_mask(self, max_len, num_cities, device):
        """
        Crea una máscara booleana de tamaño (Batch, max_len) donde True indica 
        que esa posición es padding y debe ser ignorada.
        """
        # Crea un tensor [0, 1, ..., max_len - 1] y lo expande
        idx = torch.arange(max_len, device=device).unsqueeze(0)
        # Compara con la cantidad real de ciudades para cada elemento del batch
        return idx >= num_cities.unsqueeze(1)

    # =====================================================
    # 1. --- ENCODER ---
    # =====================================================
    def encode(self, coords, visited, num_cities):
        """
        coords:     (Batch, max_cities, 2)
        visited:    (Batch, max_cities) -> Ignorado en el encoder
        num_cities: (Batch,) -> Cantidad real de ciudades
        """
        B, max_cities, _ = coords.shape
        device = coords.device

        # 1. Creamos la máscara para ignorar el padding en el Transformer
        pad_mask = self._get_pad_mask(max_cities, num_cities, device)

        # 2. Proyectamos y pasamos por el encoder
        enc_input = self.encoder_input_layer(coords)
        
        # src_key_padding_mask exige True en las posiciones que son padding
        memory = self.encoder(enc_input, src_key_padding_mask=pad_mask) 

        return memory

    # =====================================================
    # 2. --- DECODER ---
    # =====================================================
    def decode(self, memory, coords, visited, num_cities):
        """
        memory:     (Batch, max_cities, embed_dim) -> Salida del encoder
        coords:     (Batch, max_cities, 2) -> Ignorado en el decoder
        visited:    (Batch, max_cities) -> Índices ciudades visitadas (-1 para padding)
        num_cities: (Batch,) -> Cantidad real de ciudades
        """
        B, max_cities, _ = memory.shape
        device = memory.device

        # 1. --- MÁSCARA PADDING ---
        pad_mask = self._get_pad_mask(max_cities, num_cities, device)

        # 2. --- MÁSCARA CIUDADES VISITADAS ---
        visited_mask_pos = visited != -1          # (B, max_cities)

        visited_city_mask = torch.zeros(
            B, max_cities, dtype=torch.bool, device=device
        )
        batch_ids, pos_ids = visited_mask_pos.nonzero(as_tuple=True)
        visited_city_mask[batch_ids, visited[batch_ids, pos_ids]] = True

        # Máscara combinada: Prohibido atender a ciudades ya visitadas O que sean padding
        combined_mask = visited_city_mask | pad_mask

        # 3. --- DECODER: Media de ciudades visitadas ---
        # Solo usamos el contexto de las ciudades visitadas reales
        mask_ctx = visited_city_mask.unsqueeze(-1)    # (B, N, 1)

        sum_ctx = (memory * mask_ctx).sum(dim=1)
        count_ctx = mask_ctx.sum(dim=1).clamp(min=1)
        context_mean = sum_ctx / count_ctx            # (B, D)

        # 4. --- DECODER: Primera y última ciudad ---
        start_idx = visited_mask_pos.float().argmax(dim=1)
        last_idx = visited_mask_pos.sum(dim=1) - 1

        batch_idx = torch.arange(B, device=device)
        start_city_embed = memory[
            batch_idx, visited[batch_idx, start_idx].long()
        ]  # (B, D)
        
        last_city_embed = memory[
            batch_idx, visited[batch_idx, last_idx].long()
        ]   # (B, D)

        # Fusión
        ctx_concat = torch.cat(
            [context_mean, last_city_embed, start_city_embed], dim=-1
        )  # (B, 3D)
        decoder_state = self.ctx_fusion(ctx_concat)  # (B, D)

        # 5. --- DECODER: Cross-Attention (Glimpse) ---
        query = self.glimpse_proj(decoder_state).unsqueeze(1)

        for _ in range(self.num_glimpses):
            attn_out, _ = self.cross_attn(
                query=query,            
                key=memory,             
                value=memory,           
                key_padding_mask=combined_mask  # Ignora padding y visitadas
            )

            query = self.norm1(attn_out + query)   
            ff_out = self.ff(query)                
            query = self.norm2(ff_out + query)  

        attn_out = query.squeeze(1)         # (B, D)

        # 6. --- DECODER: Pointer scoring ---
        ptr_query = self.pointer_proj(attn_out)        # (B, D)

        scores = torch.matmul(
            ptr_query.unsqueeze(1),                    
            memory.transpose(1, 2)                     
        ).squeeze(1)                                   # (B, max_cities)

        scores = scores / math.sqrt(self.embed_dim)    

        # 7. --- DECODER: Masking y Softmax ---
        # Aplicamos el infinito negativo tanto a las visitadas como al padding
        scores = scores.masked_fill(combined_mask, -1e9)

        return scores