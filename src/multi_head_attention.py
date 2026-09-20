import torch
import torch.nn as nn

from src.attention import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query, key=None, value=None, mask=None):

        if key is None:
            key = query

        if value is None:
            value = key

        batch_size = query.size(0)
        query_len = query.size(1)
        key_len = key.size(1)

        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        Q = Q.view(
            batch_size,
            query_len,
            self.num_heads,
            self.head_dim
        )

        K = K.view(
            batch_size,
            key_len,
            self.num_heads,
            self.head_dim
        )

        V = V.view(
            batch_size,
            key_len,
            self.num_heads,
            self.head_dim
        )

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        attention_output, attention_weights = scaled_dot_product_attention(
            Q,
            K,
            V,
            mask
        )

        attention_output = attention_output.transpose(1, 2)

        attention_output = attention_output.contiguous().view(
            batch_size,
            query_len,
            self.d_model
        )

        output = self.out_proj(attention_output)

        return output, attention_weights