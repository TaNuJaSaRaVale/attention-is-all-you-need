import torch.nn as nn
import math

from src.multi_head_attention import MultiHeadAttention
from src.feed_forward import FeedForward
from src.positional_encoding import PositionalEncoding


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()

        self.self_attention = MultiHeadAttention(
            d_model,
            num_heads
        )

        self.feed_forward = FeedForward(
            d_model,
            d_ff
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):

        attention_output, _ = self.self_attention(
            x,
            mask=mask
        )

        x = self.norm1(
            x + attention_output
        )

        ffn_output = self.feed_forward(x)

        x = self.norm2(
            x + ffn_output
        )

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        d_ff,
        num_layers,
        max_seq_len
    ):
        super().__init__()

        self.d_model = d_model

        self.embedding = nn.Embedding(
            vocab_size,
            d_model
        )

        self.positional_encoding = PositionalEncoding(
            d_model,
            max_seq_len
        )

        self.layers = nn.ModuleList([
            EncoderLayer(
                d_model,
                num_heads,
                d_ff
            )
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):

        x = self.embedding(x)

        x = x * math.sqrt(self.d_model)

        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x, mask)

        return x