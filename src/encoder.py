import torch.nn as nn

from src.multi_head_attention import MultiHeadAttention
from src.feed_forward import FeedForward


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

    def forward(self, x):

        # Self-attention
        attention_output, _ = self.self_attention(x)

        # Residual connection + normalization
        x = self.norm1(x + attention_output)

        # Feed-forward
        ffn_output = self.feed_forward(x)

        # Residual connection + normalization
        x = self.norm2(x + ffn_output)

        return x