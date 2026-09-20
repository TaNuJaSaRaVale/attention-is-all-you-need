import torch
import torch.nn as nn

from src.encoder import Encoder
from src.decoder import Decoder


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        num_layers=6,
        max_seq_len=512
    ):
        super().__init__()

        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            max_seq_len=max_seq_len
        )

        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            max_seq_len=max_seq_len
        )

        self.output_projection = nn.Linear(
            d_model,
            tgt_vocab_size
        )

    def forward(
        self,
        src,
        tgt,
        tgt_mask=None
    ):

        encoder_output = self.encoder(src)

        decoder_output = self.decoder(
            tgt,
            encoder_output,
            tgt_mask
        )

        logits = self.output_projection(
            decoder_output
        )

        return logits