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
        self.num_heads = num_heads
        
        # Un alpha independiente POR CAPA y POR CABEZAL
        # Encoder: (num_layers, num_heads, 1, 1)
        self.alpha_enc = nn.Parameter(torch.zeros(num_encoder_layers, num_heads, 1, 1))
        # Decoder: (num_glimpses, num_heads, 1, 1)
        self.alpha_dec = nn.Parameter(torch.zeros(num_glimpses, num_heads, 1, 1))
        
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
        idx = torch.arange(max_len, device=device).unsqueeze(0)
        return idx >= num_cities.unsqueeze(1)

    # =====================================================
    # 1. --- ENCODER ---
    # =====================================================
    def encode(self, coords, distances, visited, num_cities):
        B, max_cities, _ = coords.shape
        device = coords.device

        pad_mask = self._get_pad_mask(max_cities, num_cities, device)
        memory = self.encoder_input_layer(coords)
        
        pad_mask_float = torch.zeros_like(pad_mask, dtype=memory.dtype)
        pad_mask_float = pad_mask_float.masked_fill(pad_mask, float('-inf'))

        # Iteramos capa por capa, inyectando el sesgo específico de cada nivel
        for i, layer in enumerate(self.encoder.layers):
            # Usamos el alpha correspondiente a la capa 'i'
            attn_bias = -self.alpha_enc[i] * distances.unsqueeze(1)
            attn_bias = attn_bias.view(B * self.num_heads, max_cities, max_cities)
            
            # Pasamos la memoria a través de la capa individual
            memory = layer(
                memory, 
                src_mask=attn_bias, 
                src_key_padding_mask=pad_mask_float
            )

        return memory

    # =====================================================
    # 2. --- DECODER ---
    # =====================================================
    def decode(self, memory, coords, distances, visited, num_cities):
        B, max_cities, _ = memory.shape
        device = memory.device

        pad_mask = self._get_pad_mask(max_cities, num_cities, device)
        visited_mask_pos = visited != -1          

        visited_city_mask = torch.zeros(
            B, max_cities, dtype=torch.bool, device=device
        )
        batch_ids, pos_ids = visited_mask_pos.nonzero(as_tuple=True)
        visited_city_mask[batch_ids, visited[batch_ids, pos_ids]] = True

        combined_mask = visited_city_mask | pad_mask

        mask_ctx = visited_city_mask.unsqueeze(-1)    
        sum_ctx = (memory * mask_ctx).sum(dim=1)
        count_ctx = mask_ctx.sum(dim=1).clamp(min=1)
        context_mean = sum_ctx / count_ctx            

        start_idx = visited_mask_pos.float().argmax(dim=1)
        last_idx = visited_mask_pos.sum(dim=1) - 1
        last_idx_safe = torch.clamp(last_idx, min=0)

        batch_idx = torch.arange(B, device=device)
        start_city_embed = memory[batch_idx, visited[batch_idx, start_idx].long()]  
        last_city_embed = memory[batch_idx, visited[batch_idx, last_idx_safe].long()]   

        ctx_concat = torch.cat(
            [context_mean, last_city_embed, start_city_embed], dim=-1
        )  
        decoder_state = self.ctx_fusion(ctx_concat)  

        query = self.glimpse_proj(decoder_state).unsqueeze(1) 

        last_city_idx = visited[batch_idx, last_idx_safe.long()]
        current_distances = distances[batch_idx, last_city_idx.long(), :] # (B, N)
        
        combined_mask_float = torch.zeros_like(combined_mask, dtype=memory.dtype)
        combined_mask_float = combined_mask_float.masked_fill(combined_mask, float('-inf'))

        # Iteramos el Glimpse usando el alpha específico de cada iteración
        for i in range(self.num_glimpses):
            glimpse_bias = -self.alpha_dec[i] * current_distances.unsqueeze(1).unsqueeze(2)
            glimpse_bias = glimpse_bias.view(B * self.num_heads, 1, max_cities)
            
            attn_out, _ = self.cross_attn(
                query=query,            
                key=memory,             
                value=memory,           
                key_padding_mask=combined_mask_float,
                attn_mask=glimpse_bias                 
            )

            query = self.norm1(attn_out + query)   
            ff_out = self.ff(query)                
            query = self.norm2(ff_out + query)  

        attn_out = query.squeeze(1)         

        ptr_query = self.pointer_proj(attn_out)        

        scores = torch.matmul(
            ptr_query.unsqueeze(1),                    
            memory.transpose(1, 2)                     
        ).squeeze(1)                                   

        scores = scores / math.sqrt(self.embed_dim)    
        scores = scores.masked_fill(combined_mask, -1e9)

        return scores