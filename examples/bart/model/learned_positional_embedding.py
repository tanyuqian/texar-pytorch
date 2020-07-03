import torch
from torch import nn


class LearnedPositionalEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx):
        super().__init__(num_embeddings, embedding_dim, padding_idx)

    def forward(self, input=None, positions=None):
        if positions is None:
            positions = make_positions(input, self.padding_idx)
        return super().forward(positions)

    def max_positions(self):
        return self.num_embeddings - self.padding_idx - 1


def make_positions(tensor, padding_idx):
    mask = tensor.ne(padding_idx).int()
    return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx
