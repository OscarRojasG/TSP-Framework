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
# --- 2. MÓDULO ISAB (Induced Set Attention Block) ---
# =====================================================
class ISABLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_inds=16, dropout=0.1):
        super().__init__()
        
        # Puntos inductores encapsulados exclusivamente en esta capa
        self.inducing_points = nn.Parameter(torch.randn(1, num_inds, embed_dim))
        
        # 1. INDUCE (Pack): I atiende a X
        self.induce_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm_i = nn.LayerNorm(embed_dim)

        # 2. UPDATE (Unpack): X atiende a I
        self.update_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm_x1 = nn.LayerNorm(embed_dim)

        # 3. FFN para X
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm_x2 = nn.LayerNorm(embed_dim)

    def forward(self, x, pad_mask):
        B = x.size(0)
        
        # (Batch, M, D)
        i_points = self.inducing_points.expand(B, -1, -1)

        # FASE 1: INDUCE
        i_out, _ = self.induce_attn(query=i_points, key=x, value=x, key_padding_mask=pad_mask)
        i_points = self.norm_i(i_points + i_out)

        # FASE 2: UPDATE
        x_out, _ = self.update_attn(query=x, key=i_points, value=i_points)
        x = self.norm_x1(x + x_out)

        # FASE 3: FFN
        x = self.norm_x2(x + self.ff(x))

        return x

# =====================================================
# --- 3. ARQUITECTURA PRINCIPAL (CPE + ISAB DECODER) ---
# =====================================================
class TSPTransformer(Transformer):
    def __init__(self, input_dim=2, embed_dim=128, num_heads=8, num_encoder_layers=2, num_glimpses=2, dropout_rate=0.1, p_inds=16):
        # Mantenemos las firmas originales compatibles
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
        
        # --- DECODER (Secuencial con ISAB y CPE) ---
        self.circular_pe = CircularPositionalEncoding(embed_dim=embed_dim)
        
        # Reemplazamos el Self-Attention estándar del historial por la capa ISAB
        self.tour_isab = ISABLayer(embed_dim, num_heads, num_inds=p_inds, dropout=dropout_rate)

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

        # 3. --- PROCESAMIENTO SECUENCIAL CON ISAB ---
        # Pasamos el tour al ISABLayer. Internamente usa sus propios inducing points
        tour_embeds = self.tour_isab(tour_embeds, tour_padding_mask)

        # 4. --- EXTRACCIÓN DE LA ÚLTIMA CIUDAD ---
        last_seq_idx = torch.clamp(valid_visits.sum(dim=1) - 1, min=0)
        
        # Estado del decoder: embedding de la última ciudad, enriquecido por ISAB
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