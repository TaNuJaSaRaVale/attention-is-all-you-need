import torch
from src.attention import scaled_dot_product_attention

Q = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0]
])

K = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0]
])

V = torch.tensor([
    [10.0, 20.0],
    [30.0, 40.0]
])

seq_len = Q.size(0)

mask = torch.tril(
    torch.ones(seq_len, seq_len)
)

output , weights= scaled_dot_product_attention(Q, K, V,mask)

print("Causal mask:")
print(mask)

print("Attention weights:")
print(weights)

print("Output:")
print(output)