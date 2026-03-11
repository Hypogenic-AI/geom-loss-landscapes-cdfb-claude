"""
Small transformer model for theorem proving.
Encoder-decoder architecture for sequence-to-sequence proof generation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class ProofTransformer(nn.Module):
    """
    Small transformer for proof generation.
    Uses a simple decoder-only (autoregressive) architecture where
    input and output are concatenated: [input <sep> output].
    """

    def __init__(self, vocab_size: int, d_model: int = 64, nhead: int = 2,
                 num_layers: int = 2, dim_feedforward: int = 128,
                 max_len: int = 256, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor, tgt_mask=None):
        """
        Args:
            src: [batch, seq_len] token ids (full sequence: input <sep> output)
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        seq_len = src.size(1)
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # Causal mask for autoregressive generation
        if tgt_mask is None:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=src.device)

        # Use self-attention only (decoder-only)
        # We pass the same tensor as both memory and target
        memory = torch.zeros(src.size(0), 1, self.d_model, device=src.device)
        x = self.transformer(x, memory, tgt_mask=tgt_mask)

        logits = self.output_proj(x)
        return logits

    def get_flat_params(self) -> torch.Tensor:
        """Get all parameters as a flat vector."""
        return torch.cat([p.data.view(-1) for p in self.parameters()])

    def set_flat_params(self, flat_params: torch.Tensor):
        """Set parameters from a flat vector."""
        offset = 0
        for p in self.parameters():
            numel = p.numel()
            p.data.copy_(flat_params[offset:offset + numel].view(p.shape))
            offset += numel

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def create_model(vocab_size: int, config: dict = None) -> ProofTransformer:
    """Create a model with given config."""
    default_config = {
        'd_model': 64,
        'nhead': 2,
        'num_layers': 2,
        'dim_feedforward': 128,
        'max_len': 256,
        'dropout': 0.1,
    }
    if config:
        default_config.update(config)
    return ProofTransformer(vocab_size, **default_config)


if __name__ == '__main__':
    model = create_model(vocab_size=50)
    print(f"Model parameters: {model.count_params()}")
    x = torch.randint(0, 50, (4, 32))
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
