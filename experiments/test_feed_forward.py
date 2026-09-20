import torch

from src.feed_forward import FeedForward


batch_size = 2
seq_len = 5
d_model = 512
d_ff = 2048

x = torch.randn(
    batch_size,
    seq_len,
    d_model
)

ffn = FeedForward(
    d_model=d_model,
    d_ff=d_ff
)

output = ffn(x)

print("Input shape:")
print(x.shape)

print("\nOutput shape:")
print(output.shape)


