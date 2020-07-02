import math

import torch
from torch import nn

from texar.torch import ModuleBase
from texar.torch.modules import TransformerDecoder

from .learned_positional_embedding import LearnedPositionalEmbedding


class BARTDecoder(ModuleBase):
    def __init__(self, pad_id, token_embedder, hparams=None):
        ModuleBase.__init__(self=self, hparams=hparams)

        self._token_embedder = token_embedder
        self._pos_embedder = LearnedPositionalEmbedding(
            num_embeddings=self._hparams.max_positions + pad_id + 1,
            embedding_dim=self._hparams.embedding_dim,
            padding_idx=pad_id)

        self.embed_scale = 1.0 if self._hparams.no_scale_embedding else \
            math.sqrt(self._hparams.embedding_dim)

        assert token_embedder.dim == self._hparams.embedding_dim

        if self._hparams.layernorm_embedding:
            self._layernorm_embedding = nn.LayerNorm(
                normalized_shape=token_embedder.dim,
                eps=self._hparams.transformer.eps,
                elementwise_affine=True)

        self._transformer_decoder = TransformerDecoder(
            token_pos_embedder=self._embedding_fn,
            vocab_size=token_embedder.vocab_size,
            output_layer=self._token_embedder.embedding,
            hparams=self._hparams.transformer)

    def _embedding_fn(self, tokens, _):
        x = self.embed_scale * self._token_embedder(tokens)
        x = x + self._pos_embedder(tokens)

        if self._hparams.layernorm_embedding:
            x = self._layernorm_embedding(x)

        return x

    @property
    def forward(self):
        return self._transformer_decoder.forward

    @staticmethod
    def default_hparams():
        return {
            'max_positions': 1024,
            'embedding_dim': 1024,
            'no_scale_embedding': True,
            'layernorm_embedding': True,
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
                            "kwargs": {},
                            "type": "GELU"
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
                "residual_dropout": 0.1,
                'normalize_before': False,
                'final_layer_norm': False
            }
        }
