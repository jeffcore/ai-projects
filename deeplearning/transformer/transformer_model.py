import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """
    Positional encoding for the transformer model.
    """
    def __init__(self, d_model, max_seq_length=1000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer to be part of the module's state
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, seq_length, d_model]
        Returns:
            Output tensor with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :]

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
        # Output projection
        self.output = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear projections and reshape
        q = self.query(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.key(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.value(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.output(attn_output)
        
        return output

class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class DecoderLayer(nn.Module):
    """
    Single decoder layer for the transformer.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        # Self-attention mechanism
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer normalization
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class DecoderOnlyTransformer(nn.Module):
    """
    Decoder-only transformer model.
    """
    def __init__(self, vocab_size, d_model=512, num_layers=6, num_heads=8, d_ff=2048, 
                 max_seq_length=1000, dropout=0.1):
        super(DecoderOnlyTransformer, self).__init__()
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, mask=None):
        # Get sequence length and batch size
        batch_size, seq_length = x.size()
        
        # Generate causal mask if not provided
        if mask is None:
            # Create a causal mask (lower triangular)
            mask = torch.tril(torch.ones(seq_length, seq_length)).to(x.device)
            mask = mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_length, seq_length]
        
        # Token embedding and positional encoding
        x = self.token_embedding(x) * math.sqrt(self.token_embedding.embedding_dim)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Apply decoder layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Final layer normalization
        x = self.norm(x)
        
        # Output projection
        logits = self.output_projection(x)
        
        return logits
    
    def generate(self, input_ids, max_length, temperature=1.0):
        """
        Generate text autoregressively.
        
        Args:
            input_ids: Initial input tokens [batch_size, seq_length]
            max_length: Maximum sequence length to generate
            temperature: Sampling temperature (1.0 = no change, <1.0 = more deterministic, >1.0 = more random)
        
        Returns:
            Generated token IDs
        """
        batch_size = input_ids.size(0)
        generated = input_ids
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                # Only process the last token to avoid memory issues
                # We'll use the full sequence for context but only predict the next token
                
                # Get model predictions for the entire sequence so far
                # Create a causal mask for the current sequence length
                seq_length = generated.size(1)
                mask = torch.tril(torch.ones(seq_length, seq_length)).to(generated.device)
                mask = mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_length, seq_length]
                
                # Token embedding
                x = self.token_embedding(generated) * math.sqrt(self.token_embedding.embedding_dim)
                
                # Create positional encoding manually for the exact sequence length
                d_model = x.size(-1)
                pe = torch.zeros(1, seq_length, d_model, device=generated.device)
                position = torch.arange(0, seq_length, dtype=torch.float, device=generated.device).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2, device=generated.device).float() * (-math.log(10000.0) / d_model))
                
                # Apply sine to even indices and cosine to odd indices
                pe[0, :, 0::2] = torch.sin(position * div_term)
                pe[0, :, 1::2] = torch.cos(position * div_term)
                
                # Apply positional encoding
                x = x + pe
                x = self.dropout(x)
                
                # Apply decoder layers
                for layer in self.layers:
                    x = layer(x, mask)
                
                # Final layer normalization
                x = self.norm(x)
                
                # Output projection
                logits = self.output_projection(x)
                
                # Get next token logits (last position in sequence)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Sample from the distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Concatenate with the previous tokens
                generated = torch.cat([generated, next_token], dim=1)
                
        return generated
