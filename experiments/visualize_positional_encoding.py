import matplotlib.pyplot as plt
import torch

from src.positional_encoding import PositionalEncoding


d_model = 64
seq_len = 50

pe = PositionalEncoding(
    d_model=d_model,
    max_seq_len=seq_len
)

encoding = pe.pe[0, :seq_len].detach().numpy()

plt.figure(figsize=(10, 6))
plt.imshow(encoding, aspect="auto")
plt.xlabel("Embedding Dimension")
plt.ylabel("Position")
plt.title("Sinusoidal Positional Encoding")
plt.colorbar()
plt.savefig("notes/positional_encoding.png", dpi=150, bbox_inches="tight")





encoding = pe.pe[0].detach()


plt.plot(encoding[:, 0])
plt.title("Dimension 0")
plt.xlabel("Position")
plt.ylabel("PE value")
plt.show()
plt.savefig("notes/wave.png", dpi=150, bbox_inches="tight")