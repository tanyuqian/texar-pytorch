import os

import torch

from texar.torch.modules.pretrained.pretrained_base import PretrainedMixin


class PretrainedBARTMixin(PretrainedMixin):
    _MODEL_NAME = "BART"
    _MODEL2URL = {
        'bart.large': 'https://drive.google.com/file/d/1IvBWpjjfcEK7LBIsZbw37rUiLHqnLW77/view?usp=sharing'
    }
    _MODEL2CKPT = {
        'bart.large': 'model.pt'
    }

    def __init__(self):
        pass

    def init_pretrained_weights(self, pretrained_model_name, cache_dir):
        self.pretrained_model_dir = self.download_checkpoint(
            pretrained_model_name, cache_dir)
        self._init_from_checkpoint(pretrained_model_name=pretrained_model_name,
                                   cache_dir=self.pretrained_model_dir)

    def _init_from_checkpoint(self, pretrained_model_name: str,
                              cache_dir: str, **kwargs):
        checkpoint_path = os.path.join(
            cache_dir, self._MODEL2CKPT[pretrained_model_name])
        ckpt_state_dict = torch.load(checkpoint_path)

        self.state_dict()['_token_embedder._embedding'].copy_(
            ckpt_state_dict['encoder.embed_tokens.weight'])
        self.state_dict()['_encoder._pos_embedder.weight'].copy_(
            ckpt_state_dict['encoder.embed_positions.weight'])

    def _transform_config(cls, pretrained_model_name: str,
                          cache_dir: str):
        pass


if __name__ == '__main__':
    bart_mixin = PretrainedBARTMixin()
    bart_mixin.init_pretrained_weights(
        pretrained_model_name='bart.large', cache_dir='.')