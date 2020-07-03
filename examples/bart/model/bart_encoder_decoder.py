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

        self.heads = {}
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
        self.heads[name] = nn.ModuleList()
        for i in range(len(hidden_dims)):
            u = hidden_dims[i - 1] if i != 0 else self.decoder.output_size
            v = hidden_dims[i]
            self.heads[name].extend([
                nn.Linear(u, v), nn.Dropout(self._hparams.heads_dropout)])

        u = hidden_dims[-1] if len(hidden_dims) > 0 \
            else self.decoder.output_size
        self.heads[name].append(nn.Linear(u, num_classes))

        self.add_module(name=f'head_{name}', module=self.heads[name])

        print(self.heads[name])

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
