import torch
import torch.nn as nn

from attention import AttentionBlock


class DecoderBlock(nn.Module):
    def __init__(
        self,
        embed_size: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        device: str = 'cpu',
        *args,
        **kwargs
    ) -> None:
        super(DecoderBlock, self).__init__(*args, **kwargs)
        self.selfAttention = AttentionBlock(
            embed_size=embed_size, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.crossAttention = AttentionBlock(
            embed_size=embed_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, embed_size)
        )
        self.norm3 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        key,
        value,
        src_mask=None,
        tgt_mask=None,
    ):
        self_attention = self.selfAttention(x, x, x, tgt_mask)
        query = self.dropout(self.norm1(self_attention + x))

        cross_attention = self.crossAttention(
            query=query, key=key, value=value, mask=src_mask)
        x = self.dropout(self.norm2(cross_attention + query))

        forward = self.feed_forward(x)
        x = self.dropout(self.norm3(forward + x))
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        tgt_vocab_size: int,
        embed_size: int = 512,
        num_heads: int = 8,
        num_decoder_blocks: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 100,
        device: str = 'cpu',
        *args,
        **kwargs
    ) -> None:
        super(Decoder, self).__init__(*args, **kwargs)
        self.tgt_word_embedding = nn.Embedding(tgt_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_seq_len, embed_size)

        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    embed_size=embed_size,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    device=device
                ) for i in range(num_decoder_blocks)
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(embed_size, tgt_vocab_size)

    def forward(
        self,
        x,
        encoder_output,
        src_mask=None,
        tgt_mask=None
    ):
        N, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len)

        tgt_word_embedding = self.tgt_word_embedding(x)
        tgt_position_embedding = self.position_embedding(positions)
        x = self.dropout(tgt_word_embedding + tgt_position_embedding)

        for decoder_block in self.decoder_blocks:
            x = decoder_block(x=x, key=encoder_output,
                              value=encoder_output, src_mask=src_mask, tgt_mask=tgt_mask)

        return self.fc_out(x)


if __name__ == "__main__":
    query = torch.randn(4, 10, 512)
    key = torch.randn(4, 12, 512)
    value = torch.randn(4, 12, 512)

    decoder_block = DecoderBlock(512, 8)
    out = decoder_block(
        query, key, value
    )
    print(out.shape)

    decoder = Decoder(
        5000, embed_size=512, num_decoder_blocks=6, num_heads=8
    )
    input_seq = torch.randint(0, 5000, (4, 20), dtype=torch.long)
    print(decoder(x=input_seq, encoder_output=key).shape)
