import torch
from torch import nn

from texar.torch.modules.encoder_decoders import EncoderDecoderBase
from texar.torch.modules import WordEmbedder

from .bart_tokenizer import BARTTokenizer
from .bart_encoder import BARTEncoder
from .bart_decoder import BARTDecoder
from .pretrained_bart_mixin import PretrainedBARTMixin


class BART(EncoderDecoderBase, PretrainedBARTMixin):
    def __init__(self, pretrained_model_name='bart.large', hparams=None):
        EncoderDecoderBase.__init__(self=self, hparams=hparams)

        self.tokenizer = BARTTokenizer()

        vocab_size = self.tokenizer.vocab_size
        if 'cnn' in pretrained_model_name:
            vocab_size -= 1
        self.token_embedder = WordEmbedder(
            vocab_size=vocab_size,
            hparams=self._hparams.token_embedder)

        self.encoder = BARTEncoder(
            pad_id=self.tokenizer.pad_id,
            token_embedder=self.token_embedder,
            hparams=self._hparams.encoder)

        self.decoder = BARTDecoder(
            pad_id=self.tokenizer.pad_id,
            token_embedder=self.token_embedder,
            hparams=self._hparams.decoder)

        self.heads = nn.ModuleDict()
        if 'mnli' in pretrained_model_name:
            self.register_classification_head(
                name='mnli', num_classes=3, hidden_dims=[1024])

        self.init_pretrained_weights(
            pretrained_model_name=pretrained_model_name, cache_dir='.')

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

    def extract_features(self, tokens, lengths):
        prev_output_tokens = tokens.clone()

        prev_output_tokens[:, 0] = tokens.gather(
            1, (tokens.ne(self.tokenizer.pad_id).sum(dim=1) - 1).unsqueeze(-1),
        ).squeeze()

        prev_output_tokens[:, 1:] = tokens[:, :-1]

        features = self.forward(
            src_tokens=tokens,
            src_lengths=lengths,
            decoder_input=prev_output_tokens,
            features_only=True)

        return features

    def register_classification_head(self, name, num_classes, hidden_dims):
        if name in self.heads:
            raise ValueError(f'head named {name} is already registered.')

        # dim_list = [self.decoder.output_size] + hidden_dims
        layer_list = []
        for i in range(len(hidden_dims) + 1):
            u = hidden_dims[i - 1] if i != 0 else self.decoder.output_size
            v = hidden_dims[i] if i < len(hidden_dims) else num_classes
            layer_list.extend([
                nn.Dropout(self._hparams.heads_dropout),
                nn.Linear(u, v),
                nn.Tanh()])

        self.heads[name] = nn.Sequential(*layer_list[:-1])

    def predict(self, head, tokens, lengths, return_logits=False):
        features = self.extract_features(tokens=tokens, lengths=lengths)

        sentence_representation = features[tokens.eq(
            self.tokenizer.eos_id), :].view(
            features.size(0), -1, features.size(-1))[:, -1, :]

        logits = self.heads[head](sentence_representation)

        return logits if return_logits else torch.log_softmax(logits, dim=-1)

    def generate(self, src_tokens, src_lengths,
                 beam_width=4, length_penalty=2., max_decoding_length=140):
        encoder_output = self.encoder(
            src_tokens=src_tokens, src_lengths=src_lengths)

        start_tokens = src_tokens.new_full(
            (src_tokens.shape[0],), self.vocab.bos_token_id)

        predictions = self.decoder(
            memory=encoder_output,
            memory_sequence_length=src_lengths,
            beam_width=beam_width,
            length_penalty=length_penalty,
            start_tokens=start_tokens,
            end_token=self.tokenizer.eos_id,
            max_decoding_length=max_decoding_length,
            decoding_strategy="infer_greedy",
        )

        return predictions

    @staticmethod
    def default_hparams():
        return {
            'token_embedder': {'dim': 1024},
            'loss_label_confidence': 0.9,
            'heads_dropout': 0.1,
            'encoder': None,
            'decoder': None
        }

    @property
    def encode(self):
        return self.tokenizer.encode

    @property
    def decode(self):
        return self.tokenizer.decode
