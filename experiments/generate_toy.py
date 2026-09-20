import torch

from src.transformer import Transformer


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


src_vocab_size = 20
tgt_vocab_size = 20

d_model = 64
num_heads = 4
d_ff = 256
num_layers = 2
max_seq_len = 10


model = Transformer(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    num_layers=num_layers,
    max_seq_len=max_seq_len
).to(device)


model.load_state_dict(
    torch.load(
        "transformer_toy.pth",
        map_location=device
    )
)

model.eval()


src = torch.tensor([
    [1, 2, 3]
]).to(device)


generated = torch.tensor([
    [0]
]).to(device)


with torch.no_grad():

    for _ in range(3):

        seq_len = generated.size(1)

        mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len
            )
        ).to(device)

        logits = model(
            src,
            generated,
            mask
        )

        next_token = torch.argmax(
            logits[:, -1, :],
            dim=-1,
            keepdim=True
        )

        generated = torch.cat(
            [generated, next_token],
            dim=1
        )


print("Input:")
print(src[0].tolist())

print("\nGenerated:")
print(generated[0].tolist())

print("\nPrediction:")
print(generated[0, 1:].tolist())