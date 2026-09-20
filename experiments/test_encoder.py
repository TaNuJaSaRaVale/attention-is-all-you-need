import torch

from src.encoder import EncoderLayer


batch_size = 2
seq_len = 5
d_model = 512
num_heads = 8
d_ff = 2048

x = torch.randn(
    batch_size,
    seq_len,
    d_model
)

encoder = EncoderLayer(
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff
)

output = encoder(x)

print("Input shape:")
print(x.shape)

print("\nOutput shape:")
print(output.shape)