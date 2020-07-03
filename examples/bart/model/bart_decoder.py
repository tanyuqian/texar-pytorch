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
            embedding_dim=token_embedder.dim,
            padding_idx=pad_id)

        if self._hparams.layernorm_embedding:
            self._layernorm_embedding = nn.LayerNorm(
                normalized_shape=token_embedder.dim,
                eps=self._hparams.transformer.eps,
                elementwise_affine=True)

        self._transformer_decoder = TransformerDecoder(
            token_pos_embedder=self.forward_embedding,
            vocab_size=token_embedder.vocab_size,
            output_layer=self._token_embedder.embedding,
            hparams=self._hparams.transformer)

    def forward_embedding(self, tokens, positions):
        x = self._token_embedder(tokens) + self._pos_embedder.forward(
            positions=positions + self._pos_embedder.padding_idx + 1)

        if self._hparams.layernorm_embedding:
            return self._layernorm_embedding(x)
        else:
            return x

    @property
    def forward(self):
        return self._transformer_decoder.forward

    @property
    def output_size(self):
        return self._transformer_decoder.output_size

    @staticmethod
    def default_hparams():
        return {
            'max_positions': 1024,
            'layernorm_embedding': True,
            'transformer': {
                "dim": 1024,
                "embedding_dropout": 0.1,
                "eps": 1e-5,
                "residual_dropout": 0.1,
                'normalize_before': False,
                'final_layer_norm': False,
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
                }
            }
        }
