from .learned_positional_embedding import LearnedPositionalEmbedding

from texar.torch.modules import TransformerEncoder
from texar.torch import ModuleBase


class BARTEncoder(ModuleBase):
    def __init__(self, pad_id, token_embedder, hparams=None):
        ModuleBase.__init__(self=self, hparams=hparams)

        self.token_embedder = token_embedder
        self.pos_embedder = LearnedPositionalEmbedding(
            num_embeddings=self._hparams.max_positions + pad_id + 1,
            embedding_dim=self.token_embedder.dim,
            padding_idx=pad_id)

        self._transformer_encoder = TransformerEncoder(
            hparams=self._hparams.transformer.todict())

    def forward_embedding(self, src_tokens):
        return self.token_embedder(src_tokens) + self.pos_embedder(src_tokens)

    def forward(self, src_tokens, src_lengths):
        encoder_outputs = self._transformer_encoder(
            inputs=self.forward_embedding(src_tokens),
            sequence_length=src_lengths)

        return encoder_outputs

    @staticmethod
    def default_hparams():
        return {
            'max_positions': 1024,
            'transformer': {
                "dim": 1024,
                "embedding_dropout": 0.1,
                "eps": 1e-5,
                'use_bert_config': True,
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
            }
        }

