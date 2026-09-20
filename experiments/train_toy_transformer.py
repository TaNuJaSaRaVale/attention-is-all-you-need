import torch
import torch.nn as nn

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


src = torch.tensor([
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5],
    [4, 5, 6]
]).to(device)


tgt = torch.tensor([
    [0, 4, 5, 6],
    [0, 5, 6, 7],
    [0, 6, 7, 8],
    [0, 7, 8, 9]
]).to(device)


target = torch.tensor([
    [4, 5, 6],
    [5, 6, 7],
    [6, 7, 8],
    [7, 8, 9]
]).to(device)


criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


for epoch in range(1000):

    model.train()

    optimizer.zero_grad()

    output = model(src, tgt[:, :-1])

    loss = criterion(
        output.reshape(-1, tgt_vocab_size),
        target.reshape(-1)
    )

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch {epoch + 1}, "
            f"Loss: {loss.item():.4f}"
        )


        torch.save(
    model.state_dict(),
    "transformer_toy.pth"
)

print("Model saved.")