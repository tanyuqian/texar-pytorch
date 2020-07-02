from texar.torch.modules.encoder_decoders import EncoderDecoderBase
from texar.torch.modules import WordEmbedder

from .bart_tokenizer import BARTTokenizer
from .bart_encoder import BARTEncoder
from .bart_decoder import BARTDecoder
from .label_smoothing_loss import LabelSmoothingLoss
from .pretrained_bart_mixin import PretrainedBARTMixin


class BART(EncoderDecoderBase, PretrainedBARTMixin):
    def __init__(self, pretrained_model_name='bart.large', hparams=None):
        EncoderDecoderBase.__init__(self=self, hparams=hparams)

        self.tokenizer = BARTTokenizer()

        self.token_embedder = WordEmbedder(
            vocab_size=self.tokenizer.vocab_size,
            hparams=self._hparams.token_embedder)

        self.encoder = BARTEncoder(
            pad_id=self.tokenizer.pad_id,
            token_embedder=self.token_embedder,
            hparams=self._hparams.encoder)

        self.decoder = BARTDecoder(
            pad_id=self.tokenizer.pad_id,
            token_embedder=self.token_embedder,
            hparams=self._hparams.decoder)

        self.init_pretrained_weights(
            pretrained_model_name=pretrained_model_name, cache_dir='.')

        self.smoothed_loss_func = LabelSmoothingLoss(
            label_confidence=self._hparams.loss_label_confidence,
            tgt_vocab_size=self.token_embedder.vocab_size,
            ignore_index=0)

    def forward(self, src_tokens, src_lengths, decoder_input,
                features_only=False):
        encoder_output = self.encoder(
            src_tokens=src_tokens, src_lengths=src_lengths)

        decoder_output = self.decoder(
            memory=encoder_output,
            memory_sequence_length=src_lengths,
            inputs=decoder_input,
            decoding_strategy="train_greedy",
            features_only=features_only)

        return decoder_output

    def extract_features(self, tokens):
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.size(-1) > min(self.model.max_positions()):
            raise ValueError('tokens exceeds maximum length: {} > {}'.format(
                tokens.size(-1), self.model.max_positions()
            ))
        tokens.to(device=self.device),
        prev_output_tokens = tokens.clone()

        prev_output_tokens[:, 0] = tokens.gather(
            1, (tokens.ne(self.tokenizer.pad_id).sum(dim=1) - 1).unsqueeze(-1),
        ).squeeze()

        prev_output_tokens[:, 1:] = tokens[:, :-1]

        features = self.forward(
            src_tokens=tokens,
            src_lengths=None,
            decoder_input=prev_output_tokens,
            features_only=True)

        return features

    @staticmethod
    def default_hparams():
        return {
            'token_embedder': {'dim': 1024},
            'loss_label_confidence': 0.9,
            'encoder': None,
            'decoder': None
        }

    @property
    def encode(self):
        return self.tokenizer.encode

    @property
    def decode(self):
        return self.tokenizer.decode
