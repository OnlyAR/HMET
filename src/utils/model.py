"""Define lightweight neural projection models for retrieval embeddings."""

import torch
from torch import nn


class Encoder(nn.Module):
    """Apply a single linear projection with equal input and output dimensions."""

    def __init__(self, dim: int, random_init: bool = False):
        """Initialize the linear projection.

        Args:
            dim: Input and output embedding dimension.
            random_init: Use normal initialization instead of an identity matrix.
        """
        super().__init__()
        self.dim = dim
        self.linear = nn.Linear(dim, dim, bias=False)
        if random_init:
            nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)
        else:
            nn.init.eye_(self.linear.weight)

    def forward(self, x):
        """Project a batch of embeddings."""
        x = self.linear(x)
        return x


class MultiHeadEncoder(nn.Module):
    """Average the outputs of multiple independently initialized projection heads."""

    def __init__(self, dim: int, num_heads: int = 4):
        """Initialize the projection heads.

        Args:
            dim: Input and output embedding dimension.
            num_heads: Number of linear projection heads.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.heads = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(num_heads)])
        for head in self.heads:
            nn.init.normal_(head.weight.data, mean=0.0, std=0.02)

    def forward(self, x):
        """Project embeddings and average the outputs from all heads.

        Args:
            x: Input tensor with shape ``(batch_size, dimension)``.

        Returns:
            Averaged projection tensor with the same shape as the input.
        """
        outputs = torch.stack([head(x) for head in self.heads])
        return outputs.mean(dim=0)
