import torch
import torch.nn as nn
import math
from models.base.transformer import Transformer

# =====================================================
# --- 1. MÓDULO DE POSITIONAL ENCODING CÍCLICO (CPE) ---
# =====================================================
class CircularPositionalEncoding(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        self.register_buffer('div_term', div_term)

    def forward(self, seq_len, num_cities):
        B = num_cities.shape[0]
        device = num_cities.device
        
        pos = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(0).unsqueeze(-1)
        term1 = pos * self.div_term
        
        N = num_cities.unsqueeze(1).unsqueeze(2).float()
        term2 = (2 * math.pi * pos) / N
        
        angles = term1 + term2
        
        cpe = torch.zeros(B, seq_len, self.embed_dim, device=device)
        cpe[:, :, 0::2] = torch.sin(angles)
        cpe[:, :, 1::2] = torch.cos(angles)
        
        return cpe

# =====================================================
# --- 2. MÓDULO LUNA (Linear Unified Nested Attention) ---
# =====================================================
class LUNALayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        
        self.pack_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_p = nn.LayerNorm(embed_dim)

        self.unpack_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_x1 = nn.LayerNorm(embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm_x2 = nn.LayerNorm(embed_dim)

    def forward(self, x, p, pad_mask):
        # PACK: P resume X
        p_out, _ = self.pack_attn(query=p, key=x, value=x, key_padding_mask=pad_mask)
        p = self.norm_p(p + p_out)

        # UNPACK: X recupera la información de P
        x_out, _ = self.unpack_attn(query=x, key=p, value=p)
        x = self.norm_x1(x + x_out)

        # FFN
        x = self.norm_x2(x + self.ff(x))
        return x, p

# =====================================================
# --- 3. ARQUITECTURA PRINCIPAL (CPE + LUNA DECODER) ---
# =====================================================
class TSPTransformer(Transformer):
    def __init__(self, input_dim=2, embed_dim=128, num_heads=8, num_encoder_layers=2, num_glimpses=2, dropout_rate=0.1, p_len=16):
        super().__init__(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_glimpses=num_glimpses,
            dropout_rate=dropout_rate,
        )
        self.embed_dim = embed_dim
        
        # --- ENCODER (Estándar O(N^2) porque solo se corre 1 vez) ---
        self.encoder_input_layer = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout_rate, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers, enable_nested_tensor=False)
        
        # --- DECODER (Secuencial con LUNA y CPE) ---
        self.circular_pe = CircularPositionalEncoding(embed_dim=embed_dim)
        
        # Parámetro global P para el mecanismo LUNA del historial
        self.aux_tensor = nn.Parameter(torch.randn(1, p_len, embed_dim))
        
        # Reemplazamos el Self-Attention estándar por la capa LUNA
        self.tour_luna = LUNALayer(embed_dim, num_heads, dropout_rate)

        # Glimpse / Pointer Network
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
        enc_input = self.encoder_input_layer(coords)
        
        # El Encoder global mantiene su visión completa (se ejecuta 1 sola vez)
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

        # 2. --- EXTRACCIÓN DEL TOUR E INYECCIÓN DE CPE ---
        safe_visited = visited.clone()
        safe_visited[~valid_visits] = 0

        batch_idx = torch.arange(B, device=device).unsqueeze(1)
        tour_embeds = memory[batch_idx, safe_visited.long()] # (B, max_cities, D)

        cpe = self.circular_pe(seq_len=max_cities, num_cities=num_cities)
        tour_embeds = tour_embeds + cpe

        # Máscara para ignorar posiciones vacías del tour
        tour_padding_mask = ~valid_visits
        no_visits = tour_padding_mask.all(dim=1)
        tour_padding_mask[no_visits, 0] = False 

        # 3. --- PROCESAMIENTO SECUENCIAL CON LUNA ---
        # Expandimos el tensor P global para el batch actual
        p = self.aux_tensor.expand(B, -1, -1) # (B, p_len, D)
        
        # Aplicamos LUNA: Comprime el tour histórico en P y luego lo desempaqueta en el tour
        tour_embeds, _ = self.tour_luna(tour_embeds, p, tour_padding_mask)

        # 4. --- EXTRACCIÓN DE LA ÚLTIMA CIUDAD ---
        last_seq_idx = torch.clamp(valid_visits.sum(dim=1) - 1, min=0)
        
        # El estado del decoder ahora es el embedding de la última ciudad,
        # pero ya enriquecido linealmente con toda la historia del tour gracias a LUNA
        decoder_state = tour_embeds[torch.arange(B, device=device), last_seq_idx] # (B, D)

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

        # 6. --- POINTER NETWORK ---
        ptr_query = self.pointer_proj(attn_out)        
        scores = torch.matmul(ptr_query.unsqueeze(1), memory.transpose(1, 2)).squeeze(1)                                   
        scores = scores / math.sqrt(self.embed_dim)    
        scores = scores.masked_fill(combined_mask, -1e9)

        return scores