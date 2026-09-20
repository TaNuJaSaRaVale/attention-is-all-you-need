import torch

from src.transformer import Transformer


src_vocab_size = 100
tgt_vocab_size = 100

batch_size = 2
src_seq_len = 6
tgt_seq_len = 5

model = Transformer(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    d_model=512,
    num_heads=8,
    d_ff=2048,
    num_layers=6,
    max_seq_len=50
)

src = torch.randint(
    0,
    src_vocab_size,
    (batch_size, src_seq_len)
)

tgt = torch.randint(
    0,
    tgt_vocab_size,
    (batch_size, tgt_seq_len)
)

output = model(
    src,
    tgt
)

print("Source shape:")
print(src.shape)

print("\nTarget shape:")
print(tgt.shape)

print("\nOutput shape:")
print(output.shape)