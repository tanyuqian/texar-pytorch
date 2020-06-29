from texar.torch.modules.encoder_decoders import EncoderDecoderBase
from texar.torch.modules import WordEmbedder

from .bart_tokenizer import BARTTokenizer
from .bart_encoder import BARTEncoder


class BART(EncoderDecoderBase):
    def __init__(self, hparams=None):
        EncoderDecoderBase.__init__(self=self, hparams=hparams)

        self._tokenizer = BARTTokenizer()

        self._token_embedder = WordEmbedder(vocab_size=self._hparams.vocab_size)

        self._encoder = BARTEncoder(
            pad_id=self._tokenizer.pad_id,
            token_embedder=self._token_embedder,
            hparams=self._hparams.encoder)

    @staticmethod
    def default_hparams():
        return {
            'vocab_size': 50265,
            'encoder': None,
            'decoder': None
        }