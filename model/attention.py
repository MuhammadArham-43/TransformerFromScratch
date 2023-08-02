import torch
import torch.nn as nn


class AttentionBlock(nn.Module):
    def __init__(
        self,
        embed_size: int,
        num_heads: int,
        *args,
        **kwargs
    ) -> None:
        super(AttentionBlock, self).__init__(*args, **kwargs)
        assert embed_size % num_heads == 0, 'Embed Size needs to be divisible by Heads'

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = self.embed_size // self.num_heads

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.fc_out = nn.Linear(self.embed_size, self.embed_size)

    def forward(
        self,
        query,
        key,
        value,
        mask=None
    ):
        N = query.shape[0]
        value_len, query_len, key_len = value.shape[1], query.shape[1], key.shape[1]

        # Split for Multi-head Attention
        value = value.reshape(N, value_len, self.num_heads, self.head_dim)
        key = key.reshape(N, key_len, self.num_heads, self.head_dim)
        query = query.reshape(N, query_len, self.num_heads, self.head_dim)

        # Pass through learnable linear layers
        value = self.values(value)
        key = self.keys(key)
        query = self.queries(query)

        # Change Dimension to (N, num_heads, seq_len, self.head_dim)
        value = value.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        query = query.permute(0, 2, 1, 3)

        # Flatten N and num_heads for matrix multiplication
        query = query.reshape(N * self.num_heads, query_len, self.head_dim)
        key = key.reshape(N * self.num_heads, key_len, self.head_dim)
        value = value.reshape(N * self.num_heads, value_len, self.head_dim)

        # Matrix Multiply across last two dimensions
        factor_matrix = torch.bmm(query, torch.transpose(key, 1, 2))
        factor_matrix = factor_matrix.reshape(
            N, self.num_heads, query_len, key_len)

        # Apply Masking -> Self Attention or Target Mask.
        if mask is not None:
            factor_matrix.masked_fill(mask == 0, float('-inf'))

        # Softmax across last dimension
        attention = torch.softmax(
            factor_matrix / (self.embed_size ** .5), dim=3)
        # Flatten N and num_heads again for matrix multiplication
        attention = attention.reshape(N * self.num_heads, query_len, key_len)
        attention = torch.bmm(attention, value)

        # Reverse all shapes to the original (N, seq_len, embed_dim)
        attention = attention.reshape(
            N, self.num_heads, query_len, self.head_dim)
        attention = attention.permute(0, 2, 1, 3)
        attention = attention.reshape(N, query_len, self.embed_size)

        # Pass through final linear layer
        return self.fc_out(attention)


if __name__ == "__main__":
    query = torch.randn(4, 10, 512)
    key = torch.randn(4, 12, 512)
    value = torch.randn(4, 12, 512)

    attention = AttentionBlock(512, 8)
    out = attention(
        query, key, value, mask=None
    )
    print(out.shape)
