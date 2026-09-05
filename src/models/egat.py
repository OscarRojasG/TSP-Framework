import torch
import torch.nn as nn
import math
from models.base.transformer import Transformer

# =====================================================
# --- CAPA EGAT (Edge-Augmented GAT) ---
# =====================================================

class TSP_EGATLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, alpha=0.2, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        assert embed_dim % num_heads == 0 
        self.head_dim = embed_dim // num_heads

        # Proyección de nodos
        self.W = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # --- CAMBIO 1: Proyección de la Arista (Distancia escalar -> Vector) ---
        self.W_e = nn.Linear(1, embed_dim, bias=False)
        
        # Vectores de atención (L, R y ahora E)
        self.a_L = nn.Parameter(torch.empty(1, num_heads, 1, self.head_dim))
        self.a_R = nn.Parameter(torch.empty(1, num_heads, 1, self.head_dim))
        self.a_E = nn.Parameter(torch.empty(1, num_heads, 1, 1, self.head_dim)) # Vector para aristas
        
        nn.init.xavier_uniform_(self.a_L.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_R.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_E.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm_ffn = nn.LayerNorm(embed_dim)

    def forward(self, h, edge_attr, adj_mask):
        """
        h: (B, N, D)
        edge_attr: (B, N, N, 1) Matriz de distancias
        adj_mask: (B, N, N) Máscara booleana
        """
        B, N, D = h.shape

        # 1. Proyecciones
        Wh = self.W(h).view(B, N, self.num_heads, self.head_dim).transpose(1, 2) # (B, H, N, head_dim)
        
        # Proyectamos las distancias y ajustamos dimensiones a (B, H, N, N, head_dim)
        We = self.W_e(edge_attr).view(B, N, N, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)

        # 2. Cálculo de atención
        score_L = (Wh * self.a_L).sum(dim=-1) # (B, H, N)
        score_R = (Wh * self.a_R).sum(dim=-1) # (B, H, N)
        
        # --- CAMBIO 2: Sumamos la perspectiva de la arista al score de atención ---
        score_E = (We * self.a_E).sum(dim=-1) # (B, H, N, N)
        
        scores = score_L.unsqueeze(-1) + score_R.unsqueeze(-2) + score_E # (B, H, N, N)
        e_ij = self.leakyrelu(scores)

        # 3. Enmascarado y Softmax
        _adj_mask = adj_mask.unsqueeze(1).expand_as(e_ij)
        e_ij = e_ij.masked_fill(_adj_mask, float('-inf'))
        alpha = torch.softmax(e_ij, dim=-1)
        alpha = self.dropout(alpha) # (B, H, N, N)

        # 4. Agregación (Paso de mensajes)
        # --- CAMBIO 3: El mensaje ahora es (Info del Vecino + Info de la Arista) ---
        Wh_j = Wh.unsqueeze(2) # Expandimos para broadcasting: (B, H, 1, N, head_dim)
        messages = Wh_j + We   # Suma matemática de Node + Edge: (B, H, N, N, head_dim)
        
        # Suma ponderada manual (reemplaza a torch.matmul debido a las aristas)
        h_prime = (alpha.unsqueeze(-1) * messages).sum(dim=3) # (B, H, N, head_dim)
        
        h_prime = h_prime.transpose(1, 2).contiguous().view(B, N, D)
        
        # 5. Actualización
        h = self.norm(h + self.out_proj(h_prime))
        h = self.norm_ffn(h + self.ffn(h))
        
        return h


# =====================================================
# --- ARQUITECTURA EGAT MODEL ---
# =====================================================

class TSP_EGATModel(Transformer):
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
        
        # Usamos nuestra nueva capa EGAT
        self.gat_encoder = nn.ModuleList([
            TSP_EGATLayer(embed_dim, num_heads, alpha=0.2, dropout=dropout_rate) 
            for _ in range(num_encoder_layers)
        ])
        
        # --- DECODER (Idéntico) ---
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

    def encode(self, coords, visited, num_cities, adj_matrix):
        B, max_cities, _ = coords.shape
        device = coords.device

        pad_mask = self._get_pad_mask(max_cities, num_cities, device)
        h = self.encoder_input_layer(coords)
        
        # --- CALCULAR MATRIZ DE DISTANCIAS ---
        # torch.cdist calcula eficientemente la distancia euclidiana NxN
        dist_matrix = torch.cdist(coords, coords).unsqueeze(-1) # (B, N, N, 1)
        
        adj_mask = (~adj_matrix) | pad_mask.unsqueeze(1) | pad_mask.unsqueeze(2)
        
        # Pasamos la información a través de las capas EGAT (incluyendo distancias)
        for gat_layer in self.gat_encoder:
            h = gat_layer(h, dist_matrix, adj_mask)
            
        return h
    
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