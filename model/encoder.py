import torch
import torch.nn as nn

from attention import AttentionBlock


class EncoderBlock(nn.Module):
    def __init__(
        self,
        embed_size: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        dim_feedforward: int = 2048,
        device: str = 'cpu',
        *args,
        **kwargs
    ) -> None:
        super(EncoderBlock, self).__init__(*args, **kwargs)
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.dropout = dropout

        self.attention = AttentionBlock(
            embed_size=embed_size, num_heads=num_heads)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, embed_size)
        )

    def forward(
        self,
        query,
        key,
        value,
        mask
    ):
        attention = self.attention(
            query=query, value=value, key=key, mask=mask)
        x = self.dropout(self.norm1(query + attention))
        forward = self.feed_forward(x)
        x = self.dropout(self.norm2(x + forward))
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size: int = 512,
        num_heads: int = 8,
        num_encoder_blocks: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 100,
        device='cpu',
        *args,
        **kwargs
    ) -> None:
        super(Encoder, self).__init__(*args, **kwargs)
        self.embed_size = embed_size
        self.device = device

        self.src_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.postional_embedding = nn.Embedding(max_seq_len, embed_size)

        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(
                    embed_size=embed_size,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout
                ) for i in range(num_encoder_blocks)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        N, seq_len = x.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)

        word_embedding = self.src_embedding(x)
        position_embedding = self.postional_embedding(positions)
        embedding = self.dropout(word_embedding + position_embedding)

        for encoder_block in self.encoder_blocks:
            out = encoder_block(embedding, embedding, embedding, mask)

        return out


if __name__ == "__main__":
    query = torch.randn(4, 10, 512)
    key = torch.randn(4, 12, 512)
    value = torch.randn(4, 12, 512)

    encoder_block = EncoderBlock(512, 8)
    out = encoder_block(
        query, key, value, mask=None
    )
    print(out.shape)

    encoder = Encoder(
        10000, 512, 8
    )

    input_seq = torch.randint(0, 10000, (4, 12), dtype=torch.long)
    print(input_seq.shape)
    print(encoder(input_seq).shape)
