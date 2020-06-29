from texar.torch.modules.encoder_decoders import EncoderDecoderBase
from texar.torch.modules import WordEmbedder

from .bart_tokenizer import BARTTokenizer
from .bart_encoder import BARTEncoder
from .bart_decoder import BARTDecoder
from .label_smoothing_loss import LabelSmoothingLoss


class BART(EncoderDecoderBase):
    def __init__(self, hparams=None):
        EncoderDecoderBase.__init__(self=self, hparams=hparams)

        self._tokenizer = BARTTokenizer()

        self._token_embedder = WordEmbedder(
            vocab_size=self._tokenizer.vocab_size,
            hparams=self._hparams.token_embedder)

        self._encoder = BARTEncoder(
            pad_id=self._tokenizer.pad_id,
            token_embedder=self._token_embedder,
            hparams=self._hparams.encoder)

        self._decoder = BARTDecoder(
            pad_id=self._tokenizer.pad_id,
            token_embedder=self._token_embedder,
            hparams=self._hparams.decoder)

        self.smoothed_loss_func = LabelSmoothingLoss(
            label_confidence=self._hparams.loss_label_confidence,
            tgt_vocab_size=self.vocab_size,
            ignore_index=0)

    def forward(self, src_tokens, src_lengths, decoder_input=None, labels=None,
                beam_width=None):
        encoder_output = self.encoder(
            src_tokens=src_tokens, src_lengths=src_lengths)

        if decoder_input is not None and labels is not None:
            outputs = self.decoder(
                memory=encoder_output,
                memory_sequence_length=src_lengths,
                inputs=decoder_input,
                decoding_strategy="train_greedy",
            )
            label_lengths = (labels != 0).long().sum(dim=1)
            is_target = (labels != 0).float()
            mle_loss = self.smoothed_loss_func(
                outputs.logits, labels, label_lengths)
            mle_loss = (mle_loss * is_target).sum() / is_target.sum()
            return mle_loss

        else:
            start_tokens = src_tokens.new_full(
                (src_tokens.shape[0],), self.vocab.bos_token_id)

            predictions = self.decoder(
                memory=encoder_output,
                memory_sequence_length=src_lengths,
                beam_width=beam_width,
                length_penalty=self.config_model.length_penalty,
                start_tokens=start_tokens,
                end_token=self.vocab.eos_token_id,
                max_decoding_length=self.config_data.max_decoding_length,
                decoding_strategy="infer_greedy",
            )
            # Uses the best sample by beam search
            return predictions

    @staticmethod
    def default_hparams():
        return {
            'token_embedder': {'dim': 1024},
            'loss_label_confidence': 0.9,
            'encoder': None,
            'decoder': None
        }