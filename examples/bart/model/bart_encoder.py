import torch
from torch import nn
from torch.nn import functional as F

import math

from .learned_positional_embedding import LearnedPositionalEmbedding

from texar.torch.modules import TransformerEncoder, EncoderBase

from fairseq.models.transformer import TransformerEncoder


class BARTEncoder(EncoderBase):
    def __init__(self, pad_id, token_embedder, hparams=None):
        EncoderBase.__init__(self=self, hparams=hparams)

        self._token_embedder = token_embedder
        self._pos_embedder = LearnedPositionalEmbedding(
            num_embeddings=self._hparams.max_positions + pad_id + 1,
            embedding_dim=self._hparams.embedding_dim,
            padding_idx=pad_id)

        self.embed_scale = 1.0 if self._hparams.no_scale_embedding else \
            math.sqrt(self._hparams.embedding_dim)

        if self._hparams.layernorm_embedding:
            self.layernorm_embedding = nn.LayerNorm(
                normalized_shape=self._hparams.embedding_dim,
                eps=1e-5, elementwise_affine=True)
        else:
            self.layernorm_embedding = None

        self._transformer_encoder = TransformerEncoder(
            hparams=self._hparams.transformer)

    def forward_embedding(self, src_tokens):
        # embed tokens and positions
        x = embed = self.embed_scale * self._token_embedder(src_tokens)
        if self.embed_positions is not None:
            x = embed + self._pos_embedder(src_tokens)
        return x, embed

    def forward(self, src_tokens, src_lengths):
        x, encoder_embedding = self.forward_embedding(src_tokens)

        return self._transformer_encoder(inputs=x, sequence_length=src_lengths)

    @staticmethod
    def default_hparams():
        return {
            'max_positions': 1024,
            'embedding_dim': 1024,
            'no_scale_embedding': True,
            'layernorm_embedding': True,
            'use_bert_config': True,
            'transformer': {
                "dim": 1024,
                "embedding_dropout": 0.1,
                "eps": 1e-5,
                "multihead_attention": {
                    "dropout_rate": 0.1,
                    "name": "multihead_attention",
                    "num_heads": 16,
                    "num_units": 1024,
                    "output_dim": 1024,
                    "use_bias": True
                },
                "name": "transformer_encoder",
                "num_blocks": 12,
                "poswise_feedforward": {
                    "layers": [
                        {
                            "kwargs": {
                                "bias": True,
                                "in_features": 1024,
                                "out_features": 4096
                            },
                            "type": "Linear"
                        },
                        {
                            "kwargs": {
                                "inplace": True
                            },
                            "type": "ReLU"
                        },
                        {
                            "kwargs": {
                                "p": 0.1
                            },
                            "type": "Dropout"
                        },
                        {
                            "kwargs": {
                                "bias": True,
                                "in_features": 4096,
                                "out_features": 1024
                            },
                            "type": "Linear"
                        }
                    ],
                    "name": "ffn"
                },
                "residual_dropout": 0.1
            }
        }

