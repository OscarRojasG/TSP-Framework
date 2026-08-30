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
        
        # ** PARÁMETRO DE SESGO (BIAS)
        # alpha controla qué tanto peso le damos a la distancia heurística al final
        self.alpha = nn.Parameter(torch.tensor(1.0))
        
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
        """
        coords:     (Batch, max_cities, 2)
        distances:  (Batch, max_cities, max_cities)
        visited:    (Batch, max_cities)
        num_cities: (Batch,)
        """
        B, max_cities, _ = coords.shape
        device = coords.device

        pad_mask = self._get_pad_mask(max_cities, num_cities, device)
        enc_input = self.encoder_input_layer(coords)
        
        memory = self.encoder(enc_input, src_key_padding_mask=pad_mask) 

        return memory

    # =====================================================
    # 2. --- DECODER ---
    # =====================================================
    def decode(self, memory, coords, distances, visited, num_cities):
        """
        memory:     (Batch, max_cities, embed_dim)
        coords:     (Batch, max_cities, 2)
        distances:  (Batch, max_cities, max_cities)
        visited:    (Batch, max_cities)
        num_cities: (Batch,)
        """
        B, max_cities, _ = memory.shape
        device = memory.device

        # 1. --- MÁSCARA PADDING ---
        pad_mask = self._get_pad_mask(max_cities, num_cities, device)

        # 2. --- MÁSCARA CIUDADES VISITADAS ---
        visited_mask_pos = visited != -1          

        visited_city_mask = torch.zeros(
            B, max_cities, dtype=torch.bool, device=device
        )
        batch_ids, pos_ids = visited_mask_pos.nonzero(as_tuple=True)
        visited_city_mask[batch_ids, visited[batch_ids, pos_ids]] = True

        combined_mask = visited_city_mask | pad_mask

        # 3. --- DECODER: Media de ciudades visitadas ---
        mask_ctx = visited_city_mask.unsqueeze(-1)    

        sum_ctx = (memory * mask_ctx).sum(dim=1)
        count_ctx = mask_ctx.sum(dim=1).clamp(min=1)
        context_mean = sum_ctx / count_ctx            

        # 4. --- DECODER: Primera y última ciudad ---
        start_idx = visited_mask_pos.float().argmax(dim=1)
        last_idx = visited_mask_pos.sum(dim=1) - 1
        
        # Aseguramos que el índice no sea negativo en caso de grafos vacíos
        last_idx_safe = torch.clamp(last_idx, min=0)

        batch_idx = torch.arange(B, device=device)
        start_city_embed = memory[batch_idx, visited[batch_idx, start_idx].long()]  
        last_city_embed = memory[batch_idx, visited[batch_idx, last_idx_safe].long()]   

        ctx_concat = torch.cat(
            [context_mean, last_city_embed, start_city_embed], dim=-1
        )  
        decoder_state = self.ctx_fusion(ctx_concat)  

        # 5. --- DECODER: Cross-Attention (Glimpse) ---
        query = self.glimpse_proj(decoder_state).unsqueeze(1)

        for _ in range(self.num_glimpses):
            attn_out, _ = self.cross_attn(
                query=query,            
                key=memory,             
                value=memory,           
                key_padding_mask=combined_mask  
            )

            query = self.norm1(attn_out + query)   
            ff_out = self.ff(query)                
            query = self.norm2(ff_out + query)  

        attn_out = query.squeeze(1)         

        # 6. --- DECODER: Pointer scoring ---
        ptr_query = self.pointer_proj(attn_out)        

        scores = torch.matmul(
            ptr_query.unsqueeze(1),                    
            memory.transpose(1, 2)                     
        ).squeeze(1)                                   

        scores = scores / math.sqrt(self.embed_dim)    

        # =====================================================
        # ** 7. --- APLICACIÓN DEL SESGO (BIAS) ---
        # =====================================================
        # Obtenemos los índices de las últimas ciudades visitadas
        last_city_idx = visited[batch_idx, last_idx_safe.long()]
        
        # Extraemos la distancia desde la ciudad actual a todas las demás
        # current_distances shape: (B, max_cities)
        current_distances = distances[batch_idx, last_city_idx.long(), :]
        
        # Definimos phi(i) como la distancia negativa.
        # Sumamos alpha * phi, lo que equivale a restar alpha * distancia.
        # A mayor distancia, menor será el score final.
        phi = -current_distances
        scores = scores + (self.alpha * phi)

        # 8. --- DECODER: Masking y Softmax ---
        scores = scores.masked_fill(combined_mask, -1e9)

        return scores