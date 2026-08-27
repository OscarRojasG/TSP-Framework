import torch
import torch.nn as nn
import math
from models.base.transformer import Transformer

class SpatialPositionalEncoding(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        
        # We split the dimensions in half: one half for X, one half for Y
        self.half_dim = embed_dim // 2
        
        # Division term for the frequencies
        div_term = torch.exp(
            torch.arange(0, self.half_dim, 2).float() * (-math.log(10000.0) / self.half_dim)
        )
        # We use register_buffer so it moves to the GPU automatically but doesn't require gradients
        self.register_buffer('div_term', div_term)

    def forward(self, coords):
        """
        coords: (Batch, N, 2)
        Returns: (Batch, N, embed_dim)
        """
        B, N, _ = coords.shape
        
        # Separate X and Y coordinates
        x = coords[:, :, 0].unsqueeze(-1)  # (Batch, N, 1)
        y = coords[:, :, 1].unsqueeze(-1)  # (Batch, N, 1)
        
        # Initialize the positional encoding tensors
        pe_x = torch.zeros(B, N, self.half_dim, device=coords.device)
        pe_y = torch.zeros(B, N, self.half_dim, device=coords.device)
        
        # Apply Sin to even indices and Cos to odd indices
        pe_x[:, :, 0::2] = torch.sin(x * self.div_term)
        pe_x[:, :, 1::2] = torch.cos(x * self.div_term)
        
        pe_y[:, :, 0::2] = torch.sin(y * self.div_term)
        pe_y[:, :, 1::2] = torch.cos(y * self.div_term)
        
        # Concatenate X and Y encodings to get the full embed_dim
        pe_spatial = torch.cat([pe_x, pe_y], dim=-1)
        
        return pe_spatial


class TSPTransformer(Transformer):
    def __init__(self, embed_dim=128, num_heads=8, num_encoder_layers=2, num_glimpses=2, dropout_rate=0.1):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_glimpses=num_glimpses,
            dropout_rate=dropout_rate,
        )
        self.embed_dim = embed_dim
        
        # =====================================================
        # --- ENCODER ---
        # =====================================================
        self.spatial_pe = SpatialPositionalEncoding(embed_dim=embed_dim)
        
        # Projects the spatial embeddings to allow the network to adjust them
        self.encoder_input_layer = nn.Linear(embed_dim, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dropout=dropout_rate,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers, enable_nested_tensor=False)
        
        # =====================================================
        # --- DECODER ---
        # =====================================================
        # Fusion layer for: [context_mean, last_city, start_city]
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
        Creates a boolean mask (Batch, max_len) where True indicates padding.
        """
        idx = torch.arange(max_len, device=device).unsqueeze(0)
        return idx >= num_cities.unsqueeze(1)

    # =====================================================
    # 1. --- ENCODE ---
    # =====================================================
    def encode(self, coords, visited, num_cities):
        """
        coords:     (Batch, max_cities, 2)
        visited:    Ignored in the encoder
        num_cities: (Batch,)
        """
        B, max_cities, _ = coords.shape
        device = coords.device

        pad_mask = self._get_pad_mask(max_cities, num_cities, device)

        # Apply Spatial Positional Encoding to coordinates
        spatial_embeds = self.spatial_pe(coords)
        
        # Project them before the Transformer
        enc_input = self.encoder_input_layer(spatial_embeds)
        
        # Pass through the Transformer Encoder
        memory = self.encoder(enc_input, src_key_padding_mask=pad_mask) 

        return memory

    # =====================================================
    # 2. --- DECODE ---
    # =====================================================
    def decode(self, memory, coords, visited, num_cities):
        """
        memory:     (Batch, max_cities, embed_dim) -> Encoder output
        coords:     Ignored in the decoder
        visited:    (Batch, max_cities) -> Visited cities indices (-1 for padding)
        num_cities: (Batch,)
        """
        B, max_cities, _ = memory.shape
        device = memory.device

        # 1. --- MASKS ---
        pad_mask = self._get_pad_mask(max_cities, num_cities, device)

        visited_mask_pos = visited != -1  # True for valid steps

        visited_city_mask = torch.zeros(
            B, max_cities, dtype=torch.bool, device=device
        )
        batch_ids, pos_ids = visited_mask_pos.nonzero(as_tuple=True)
        visited_city_mask[batch_ids, visited[batch_ids, pos_ids]] = True

        # Combined mask: Forbidden to attend visited cities OR padding
        combined_mask = visited_city_mask | pad_mask

        # 2. --- DECODER: Context Mean ---
        mask_ctx = visited_city_mask.unsqueeze(-1)    # (B, N, 1)

        sum_ctx = (memory * mask_ctx).sum(dim=1)
        count_ctx = mask_ctx.sum(dim=1).clamp(min=1)
        context_mean = sum_ctx / count_ctx            # (B, D)

        # 3. --- DECODER: Start & Last City ---
        start_idx = visited_mask_pos.float().argmax(dim=1)
        # Clamped to min=0 to avoid -1 indexing if no cities are visited yet
        last_idx = torch.clamp(visited_mask_pos.sum(dim=1) - 1, min=0)

        batch_idx = torch.arange(B, device=device)
        start_city_embed = memory[
            batch_idx, visited[batch_idx, start_idx].long()
        ]  # (B, D)
        
        last_city_embed = memory[
            batch_idx, visited[batch_idx, last_idx].long()
        ]   # (B, D)

        # 4. --- DECODER: Fusion ---
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
                key_padding_mask=combined_mask  
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

        # 7. --- DECODER: Final Masking ---
        scores = scores.masked_fill(combined_mask, -1e9)

        return scores