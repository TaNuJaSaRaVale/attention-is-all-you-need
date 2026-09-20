import torch

from src.decoder import DecoderLayer


batch_size = 2
decoder_seq_len = 5
encoder_seq_len = 7
d_model = 512
num_heads = 8
d_ff = 2048


decoder_input = torch.randn(
    batch_size,
    decoder_seq_len,
    d_model
)

encoder_output = torch.randn(
    batch_size,
    encoder_seq_len,
    d_model
)

decoder = DecoderLayer(
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff
)

output = decoder(
    decoder_input,
    encoder_output
)

print("Decoder input:")
print(decoder_input.shape)

print("\nEncoder output:")
print(encoder_output.shape)

print("\nDecoder output:")
print(output.shape)