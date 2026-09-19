import torch

from src.multi_head_attention import MultiHeadAttention


batch_size = 1
seq_len = 3
d_model = 8
num_heads = 2

x = torch.randn(batch_size, seq_len, d_model)

mha = MultiHeadAttention(
    d_model=d_model,
    num_heads=num_heads
)

output, weights = mha(x)

print("Input shape:")
print(x.shape)

print("\nOutput shape:")
print(output.shape)

print("\nAttention weights shape:")
print(weights.shape)