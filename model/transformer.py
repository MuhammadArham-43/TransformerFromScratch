import torch
import torch.nn as nn

from encoder import Encoder
from decoder import Decoder


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_pad_idx: int,
        tgt_pad_idx: int,
        embed_size: int = 512,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        device: str = 'cpu',
        max_seq_len: int = 100,
        *args,
        **kwargs
    ) -> None:
        super(Transformer, self).__init__(*args, **kwargs)

        self.encoder = Encoder(
            src_vocab_size=src_vocab_size,
            embed_size=embed_size,
            num_heads=num_heads,
            num_encoder_blocks=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=max_seq_len,
            device=device
        )

        self.decoder = Decoder(
            tgt_vocab_size=tgt_vocab_size,
            embed_size=embed_size,
            num_heads=num_heads,
            num_decoder_blocks=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            device=device
        )

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.device = device

    def make_src_mask(self, src_seq):
        src_mask = (src_seq != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_tgt_mask(self, tgt_seq):
        N, tgt_len = tgt_seq.shape
        tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len)
                              ).expand(N, 1, tgt_len, tgt_len)
        return tgt_mask.to(self.device)

    def forward(self, src_seq, tgt_seq):
        src_mask = self.make_src_mask(src_seq)
        tgt_mask = self.make_tgt_mask(tgt_seq)
        enc_out = self.encoder(x=src_seq, mask=src_mask)
        output = self.decoder(x=tgt_seq, encoder_output=enc_out,
                              src_mask=src_mask, tgt_mask=tgt_mask)
        return output


if __name__ == "__main__":
    model = Transformer(10000, 8000, 0, 0)
    src_seq = torch.tensor(
        [
            [1, 2, 64, 21, 135, 1135, 1231, 1256, 0, 0, 0, 0],
            [214, 124, 12, 15, 21, 51, 53, 152, 0, 0, 0, 0]
        ]
    )

    tgt_seq = torch.tensor(
        [
            [4, 1, 2, 11, 2412, 0, 0, 0],
            [4, 123, 31, 12, 124, 421, 1, 0]
        ]
    )

    src_pad_idx = tgt_pad_idx = 0
    criterion = torch.nn.CrossEntropyLoss()
    output = model(src_seq, tgt_seq[:, :-1])
    print(tgt_seq.shape)
    print(output.shape)
    print(criterion(output.reshape(-1, 8000),
          tgt_seq[:, :-1].reshape(-1)).item())
