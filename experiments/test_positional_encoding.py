import torch

from src.positional_encoding import PositionalEncoding


d_model = 8
seq_len = 5

x = torch.zeros(1, seq_len, d_model)

positional_encoding = PositionalEncoding(
    d_model=d_model,
    max_seq_len=100
)

output = positional_encoding(x)

print("Input shape:")
print(x.shape)

print("\nOutput shape:")
print(output.shape)

print("\nPositional encoding:")
print(positional_encoding.pe[0, :seq_len])