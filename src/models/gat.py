import torch
import torch.nn as nn
import math
from models.base.transformer import Transformer

# =====================================================
# --- CAPA GAT PURA (Basada en Veličković et al. 2018) ---
# =====================================================

class TSP_GATLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, alpha=0.2, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Debe ser divisible
        assert embed_dim % num_heads == 0 
        self.head_dim = embed_dim // num_heads

        # 1. Proyección Lineal de las características (W)
        self.W = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # 2. Vectores de atención a_L y a_R (separados para optimización de memoria O(N))
        self.a_L = nn.Parameter(torch.empty(1, num_heads, 1, self.head_dim))
        self.a_R = nn.Parameter(torch.empty(1, num_heads, 1, self.head_dim))
        nn.init.xavier_uniform_(self.a_L.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_R.data, gain=1.414)

        # Activaciones y regularización originales del paper
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        
        # Proyección de salida, Conexión Residual y Normalización (Esenciales para TSP)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.norm = nn.LayerNorm(embed_dim)
        
        # Red Feed-Forward (MLP) estándar en bloques GNN/Transformer modernos
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm_ffn = nn.LayerNorm(embed_dim)

    def forward(self, h, adj_mask):
        """
        h: (B, N, D)
        adj_mask: (B, N, N) Máscara booleana. True = NO hay arista / Es padding.
        """
        B, N, D = h.shape

        # --- 1. PROYECCIÓN LINEAL (W * h) ---
        # Proyectamos y separamos en múltiples cabezales: (B, H, N, head_dim)
        Wh = self.W(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # --- 2. CÁLCULO DE ATENCIÓN (a^T [Wh_i || Wh_j]) ---
        # Aplicamos el truco algebraico de PyTorch Geometric
        score_L = (Wh * self.a_L).sum(dim=-1) # (B, H, N)
        score_R = (Wh * self.a_R).sum(dim=-1) # (B, H, N)
        
        # Broadcasting para armar la matriz NxN: e_ij = score_i + score_j
        scores = score_L.unsqueeze(-1) + score_R.unsqueeze(-2) # (B, H, N, N)
        
        # LeakyReLU original de Veličković
        e_ij = self.leakyrelu(scores)

        # --- 3. ENMASCARADO ESPARSO ---
        # Bloqueamos la atención donde no existe arista en el grafo
        _adj_mask = adj_mask.unsqueeze(1).expand_as(e_ij) # Expandir cabezales
        e_ij = e_ij.masked_fill(_adj_mask, float('-inf'))

        # --- 4. AGREGACIÓN (Softmax) ---
        alpha = torch.softmax(e_ij, dim=-1)
        alpha = self.dropout(alpha)

        # Suma ponderada de los vecinos: (B, H, N, N) x (B, H, N, head_dim)
        h_prime = torch.matmul(alpha, Wh) # (B, H, N, head_dim)
        
        # Concatenación de cabezales (B, N, D)
        h_prime = h_prime.transpose(1, 2).contiguous().view(B, N, D)
        
        # --- 5. ACTUALIZACIÓN (Update con Residual & FFN) ---
        # Conexión residual 1
        h = self.norm(h + self.out_proj(h_prime))
        
        # Conexión residual 2 (Feed Forward Network)
        h = self.norm_ffn(h + self.ffn(h))
        
        return h


# =====================================================
# --- ARQUITECTURA GAT MODEL (ENCODER GAT PURO) ---
# =====================================================

class TSP_GATModel(Transformer):
    def __init__(self, input_dim=2, embed_dim=128, num_heads=8, num_encoder_layers=4, num_glimpses=2, dropout_rate=0.1):
        super().__init__(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_glimpses=num_glimpses,
            dropout_rate=dropout_rate,
        )
        self.embed_dim = embed_dim
        
        self.encoder_input_layer = nn.Linear(input_dim, embed_dim)
        
        # El estado del arte en GNN para ruteo apila entre 4 y 6 capas.
        self.gat_encoder = nn.ModuleList([
            TSP_GATLayer(embed_dim, num_heads, alpha=0.2, dropout=dropout_rate) 
            for _ in range(num_encoder_layers)
        ])
        
        # --- DECODER ---
        self.ctx_fusion = nn.Linear(3 * embed_dim, embed_dim)
        self.num_glimpses = num_glimpses
        self.glimpse_proj = nn.Linear(embed_dim, embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(nn.Linear(embed_dim, 4 * embed_dim), nn.ReLU(), nn.Linear(4 * embed_dim, embed_dim))
        self.norm2 = nn.LayerNorm(embed_dim)
        self.pointer_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def _get_pad_mask(self, max_len, num_cities, device):
        idx = torch.arange(max_len, device=device).unsqueeze(0)
        return idx >= num_cities.unsqueeze(1)

    # =====================================================
    # 1. --- ENCODE ---
    # =====================================================
    def encode(self, coords, visited, num_cities, adj_matrix):
        B, max_cities, _ = coords.shape
        device = coords.device

        pad_mask = self._get_pad_mask(max_cities, num_cities, device)
        h = self.encoder_input_layer(coords)
        
        # Construimos la máscara del grafo: Bloquear si no hay arista O si es nodo padding
        adj_mask = (~adj_matrix) | pad_mask.unsqueeze(1) | pad_mask.unsqueeze(2)
        
        # Pasamos la información a través del grafo esparso N veces (N saltos)
        for gat_layer in self.gat_encoder:
            h = gat_layer(h, adj_mask)
            
        return h

    # =====================================================
    # 2. --- DECODER ---
    # =====================================================
    def decode(self, memory, coords, visited, num_cities, adj_matrix):
        B, max_cities, _ = memory.shape
        device = memory.device

        # 1. Máscaras iniciales
        pad_mask = self._get_pad_mask(max_cities, num_cities, device)
        visited_mask_pos = visited != -1

        visited_city_mask = torch.zeros(
            B, max_cities, dtype=torch.bool, device=device
        )
        batch_ids, pos_ids = visited_mask_pos.nonzero(as_tuple=True)
        visited_city_mask[batch_ids, visited[batch_ids, pos_ids]] = True

        # 2. Máscara combinada (ignoramos adj_matrix en la salida)
        combined_mask = visited_city_mask | pad_mask

        # 3. Contexto Decoder
        mask_ctx = visited_city_mask.unsqueeze(-1)
        sum_ctx = (memory * mask_ctx).sum(dim=1)
        count_ctx = mask_ctx.sum(dim=1).clamp(min=1)
        context_mean = sum_ctx / count_ctx            

        start_idx = visited_mask_pos.float().argmax(dim=1)
        last_idx_valid = torch.clamp(visited_mask_pos.sum(dim=1) - 1, min=0)
        batch_idx = torch.arange(B, device=device)

        start_city_embed = memory[batch_idx, visited[batch_idx, start_idx].long()]
        last_city_embed = memory[batch_idx, visited[batch_idx, last_idx_valid].long()]   

        ctx_concat = torch.cat([context_mean, last_city_embed, start_city_embed], dim=-1)
        decoder_state = self.ctx_fusion(ctx_concat) 

        # 4. Glimpse
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

        # 5. Pointer scoring
        ptr_query = self.pointer_proj(attn_out)        
        scores = torch.matmul(ptr_query.unsqueeze(1), memory.transpose(1, 2)).squeeze(1)                                   
        scores = scores / math.sqrt(self.embed_dim)    

        # 6. Salida
        scores = scores.masked_fill(combined_mask, -1e9)

        return scores