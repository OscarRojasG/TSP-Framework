import torch
import torch.nn as nn
import math
from models.base.transformer import Transformer

# =====================================================
# --- MÓDULO DE POSITIONAL ENCODING CÍCLICO ---
# =====================================================

class CircularPositionalEncoding(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        # El término de división clásico (10000^(2i/d_model))
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        self.register_buffer('div_term', div_term)

    def forward(self, seq_len, num_cities):
        """
        seq_len: Largo máximo de la secuencia (max_cities para cubrir el padding)
        num_cities: Tensor (Batch,) con el valor real de N para cada instancia
        """
        B = num_cities.shape[0]
        device = num_cities.device
        
        # pos: (1, seq_len, 1)
        pos = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(0).unsqueeze(-1)
        
        # term1: pos / 10000^(2i/d_model) -> (1, seq_len, embed_dim/2)
        term1 = pos * self.div_term
        
        # term2: 2 * pi * pos / N -> (Batch, seq_len, 1)
        N = num_cities.unsqueeze(1).unsqueeze(2).float()
        term2 = (2 * math.pi * pos) / N
        
        # angles: (Batch, seq_len, embed_dim/2)
        angles = term1 + term2
        
        cpe = torch.zeros(B, seq_len, self.embed_dim, device=device)
        cpe[:, :, 0::2] = torch.sin(angles)
        cpe[:, :, 1::2] = torch.cos(angles)
        
        return cpe

# =====================================================
# --- ARQUITECTURA PRINCIPAL ---
# =====================================================

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
        
        # --- ENCODER (Estándar) ---
        # Proyectamos directamente las coordenadas 2D al embed_dim
        self.encoder_input_layer = nn.Linear(input_dim, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout_rate, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers, enable_nested_tensor=False)
        
        # --- DECODER (Secuencial con PE Circular) ---
        self.circular_pe = CircularPositionalEncoding(embed_dim=embed_dim)
        
        # Capa de Self-Attention para procesar el tour histórico
        self.tour_self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout_rate, batch_first=True
        )
        self.tour_norm = nn.LayerNorm(embed_dim)

        self.num_glimpses = num_glimpses
        self.glimpse_proj = nn.Linear(embed_dim, embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout_rate, batch_first=True
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
    def encode(self, coords, visited, num_cities):
        B, max_cities, _ = coords.shape
        device = coords.device

        pad_mask = self._get_pad_mask(max_cities, num_cities, device)
        
        # Entrada directa de las coordenadas crudas (sin Spatial PE)
        enc_input = self.encoder_input_layer(coords)
        
        memory = self.encoder(enc_input, src_key_padding_mask=pad_mask) 
        return memory

    # =====================================================
    def decode(self, memory, coords, visited, num_cities):
        B, max_cities, _ = memory.shape
        device = memory.device

        # 1. --- MÁSCARAS GENERALES ---
        pad_mask = self._get_pad_mask(max_cities, num_cities, device)
        valid_visits = (visited != -1)

        visited_city_mask = torch.zeros(B, max_cities, dtype=torch.bool, device=device)
        batch_ids, pos_ids = valid_visits.nonzero(as_tuple=True)
        visited_city_mask[batch_ids, visited[batch_ids, pos_ids]] = True

        combined_mask = visited_city_mask | pad_mask

        # 2. --- DECODER: Procesamiento Secuencial del Tour ---
        safe_visited = visited.clone()
        safe_visited[~valid_visits] = 0

        batch_idx = torch.arange(B, device=device).unsqueeze(1)
        # Extraemos los embeddings de las ciudades en orden de visita: (Batch, max_cities, D)
        tour_embeds = memory[batch_idx, safe_visited.long()] 

        # Generamos y sumamos el Circular Positional Encoding
        cpe = self.circular_pe(seq_len=max_cities, num_cities=num_cities)
        tour_embeds = tour_embeds + cpe

        # Máscara para que el Self-Attention ignore las posiciones de padding del tour
        tour_padding_mask = ~valid_visits
        
        # Truco de estabilidad: Prevención de NaNs al inicio del episodio
        no_visits = tour_padding_mask.all(dim=1)
        tour_padding_mask[no_visits, 0] = False 

        # Self-Attention sobre la secuencia histórica
        attn_tour, _ = self.tour_self_attn(
            query=tour_embeds, 
            key=tour_embeds, 
            value=tour_embeds, 
            key_padding_mask=tour_padding_mask
        )
        
        # Conexión residual y normalización
        tour_embeds = self.tour_norm(tour_embeds + attn_tour)

        # 3. --- EXTRACCIÓN DE LA ÚLTIMA CIUDAD (El Representante del Tour) ---
        # Encontramos el índice de la última ciudad visitada en la secuencia
        last_seq_idx = torch.clamp(valid_visits.sum(dim=1) - 1, min=0)
        
        # Extraemos solo el vector final que ya procesó toda la historia
        decoder_state = tour_embeds[torch.arange(B, device=device), last_seq_idx] # (B, D)

        # 4. --- DECODER: Cross-Attention (Glimpse) ---
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

        # 5. --- DECODER: Pointer scoring & Masking ---
        ptr_query = self.pointer_proj(attn_out)        
        scores = torch.matmul(ptr_query.unsqueeze(1), memory.transpose(1, 2)).squeeze(1)                                   
        scores = scores / math.sqrt(self.embed_dim)    
        scores = scores.masked_fill(combined_mask, -1e9)

        return scores