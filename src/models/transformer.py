import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TSPTransformer(nn.Module):
    def __init__(self, input_dim=2, embed_dim=128, num_heads=8, num_layers=2, dropout_rate=0.1):
        super(TSPTransformer, self).__init__()
        self.embed_dim = embed_dim
        
        # --- ENCODER ---
        # Proyección inicial de coordenadas (x, y)
        self.encoder_input_layer = nn.Linear(input_dim, embed_dim)
        
        # Transformer Encoder (Sin Positional Encoding)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dropout=dropout_rate,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # --- DECODER ---
        # Positional Encoding para el orden de visita
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=50)
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dropout=dropout_rate,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, x_src, visited):
        """
        x_src: (batch, num_cities, 2) -> Coordenadas
        visited: (batch, num_cities) -> Índice ciudades visitadas (-1 para padding)
        """
        B, num_cities, _ = x_src.shape

        # 1. --- ENCODER ---
        # Proyectamos las coordenadas y pasamos por el encoder
        enc_input = self.encoder_input_layer(x_src)
        memory = self.encoder(enc_input) # (batch, n_cities, embed_dim)

        # 2. --- DECODER ---
        # Tomamos los embeddings de las ciudades que YA hemos visitado
        batch_contexts = []
        for b in range(B):
            valid_mask = (visited[b] != -1)

            # Seleccionar embeddings de las ciudades visitadas
            ctx = memory[b, visited[b, valid_mask].long()]  # (N_context, d_model)
            # aplicar PE a las ciudades visitadas
            ctx = self.pos_encoder(ctx.unsqueeze(0)).squeeze(0)  # (N_context, d_model)

            batch_contexts.append(ctx)

        # Aplicar padding a las secuencias de contexto para formar un batch rectangular
        max_ctx = max(c.shape[0] for c in batch_contexts)
        x_tgt = torch.zeros((B, max_ctx, self.embed_dim))
        tgt_key_padding_mask = torch.ones((B, max_ctx), dtype=torch.bool)
        for b, ctx in enumerate(batch_contexts):
            n = ctx.shape[0]
            x_tgt[b, :n] = ctx
            tgt_key_padding_mask[b, :n] = False  # False = posición válida

        # --- Cross-attention ---
        decoder_out = self.decoder(
            tgt=x_tgt,
            memory=memory,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        # 3. --- PREDICCIÓN DEL SIGUIENTE PASO ---
        # Solo nos interesa la salida del ÚLTIMO paso de la secuencia temporal
        # para predecir qué viene después.
        last_step_output = decoder_out[:, -1, :] # (batch, embed_dim)

        # Calculamos atención (compatibilidad) contra todas las ciudades de la memoria
        # (batch, embed_dim) @ (batch, embed_dim, n_cities) -> (batch, n_cities)
        logits = torch.matmul(last_step_output.unsqueeze(1), memory.transpose(1, 2)).squeeze(1)
        
        # Scaling
        logits = logits / math.sqrt(self.encoder_input_layer.out_features)

        # 4. --- ENMASCARADO DE VISITADOS ---
        # Debemos prohibir volver a visitar ciudades que ya están en 'visited'.
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(1, visited.clamp(min=0), visited >= 0)
        mask[:,0] = True
        logits.masked_fill_(mask, float('-inf'))

        # 5. --- SOFTMAX ---
        probs = F.softmax(logits, dim=-1)
        
        return probs

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]