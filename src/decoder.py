import torch.nn as nn

from src.multi_head_attention import MultiHeadAttention
from src.feed_forward import FeedForward


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()

        self.self_attention = MultiHeadAttention(
            d_model,
            num_heads
        )

        self.cross_attention = MultiHeadAttention(
            d_model,
            num_heads
        )

        self.feed_forward = FeedForward(
            d_model,
            d_ff
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, mask=None):

        # Masked self-attention
        self_attention_output, _ = self.self_attention(
            x,
            mask
        )

        x = self.norm1(
            x + self_attention_output
        )

        # Encoder-decoder cross-attention
        cross_attention_output, _ = self.cross_attention(
            x,
            encoder_output
        )

        x = self.norm2(
            x + cross_attention_output
        )

        # Feed-forward network
        ffn_output = self.feed_forward(x)

        x = self.norm3(
            x + ffn_output
        )

        return x