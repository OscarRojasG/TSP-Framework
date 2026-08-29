import torch
import torch.nn as nn
import math
from models.base.transformer import Transformer

# =====================================================
# --- MÓDULO ISAB (Induced Set Attention Block) ---
# =====================================================

class ISABLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_inds=16, dropout=0.1):
        super().__init__()
        
        # ** DIFERENCIA CLAVE CON LUNA: Los puntos inductores viven DENTRO de la capa
        # Son parámetros aprendibles específicos y exclusivos de esta capa.
        self.inducing_points = nn.Parameter(torch.randn(1, num_inds, embed_dim))
        
        # 1. INDUCE (Pack): Los puntos inductores (I) atienden a la secuencia (X)
        self.induce_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm_i = nn.LayerNorm(embed_dim)

        # 2. UPDATE (Unpack): La secuencia (X) atiende al resumen inducido (I)
        self.update_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm_x1 = nn.LayerNorm(embed_dim)

        # 3. Feed Forward para X
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm_x2 = nn.LayerNorm(embed_dim)

    def forward(self, x, pad_mask):
        """
        x: (Batch, N, D) - Secuencia original
        pad_mask: (Batch, N) - Máscara de padding de X
        """
        B = x.size(0)
        
        # Expandimos los puntos inductores internos al tamaño del Batch
        # i_points: (Batch, M, D)
        i_points = self.inducing_points.expand(B, -1, -1)

        # --- FASE 1: INDUCE (Resumir) ---
        # Query = I, Key/Value = X. 
        i_out, _ = self.induce_attn(
            query=i_points, 
            key=x, 
            value=x, 
            key_padding_mask=pad_mask
        )
        i_points = self.norm_i(i_points + i_out)

        # --- FASE 2: UPDATE (Actualizar) ---
        # Query = X, Key/Value = I. 
        # No hay máscara de padding para i_points porque M es fijo.
        x_out, _ = self.update_attn(
            query=x, 
            key=i_points, 
            value=i_points
        )
        x = self.norm_x1(x + x_out)

        # --- FASE 3: FEED FORWARD ---
        x = self.norm_x2(x + self.ff(x))

        # ** DIFERENCIA CLAVE: Solo retornamos X. 
        # i_points muere aquí (no se propaga a la siguiente capa).
        return x


# =====================================================
# --- ARQUITECTURA PRINCIPAL ---
# =====================================================

class TSPTransformer(Transformer):
    # Cambiamos p_len por m_inds (cantidad de inducing points M)
    def __init__(self, input_dim=2, embed_dim=128, num_heads=8, num_encoder_layers=2, num_glimpses=2, dropout_rate=0.1, m_inds=16):
        super().__init__(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_glimpses=num_glimpses,
            dropout_rate=dropout_rate,
            m_inds=m_inds
        )
        self.embed_dim = embed_dim
        
        # --- ENCODER (ISAB) ---
        self.encoder_input_layer = nn.Linear(input_dim, embed_dim)
        
        # Apilamos las capas ISAB. 
        # Ya no hay parámetro global aquí, cada capa se encarga de sus propios puntos.
        self.isab_layers = nn.ModuleList([
            ISABLayer(embed_dim, num_heads, num_inds=m_inds, dropout=dropout_rate) 
            for _ in range(num_encoder_layers)
        ])
        
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
    def encode(self, coords, visited, num_cities):
        B, max_cities, _ = coords.shape
        device = coords.device

        pad_mask = self._get_pad_mask(max_cities, num_cities, device)
        x = self.encoder_input_layer(coords)
        
        # El bucle es idéntico a un Transformer tradicional.
        # No pasamos ni recolectamos ningún tensor auxiliar.
        for layer in self.isab_layers:
            x = layer(x, pad_mask)
            
        return x

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