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

        no_cond_map = {
            'token_embedder._embedding': 'encoder.embed_tokens.weight',
            'encoder.pos_embedder.weight': 'encoder.embed_positions.weight',
            'encoder._transformer_encoder.input_normalizer.weight':
                'encoder.layernorm_embedding.weight',
            'encoder._transformer_encoder.input_normalizer.bias':
                'encoder.layernorm_embedding.bias',
        }

        layer_c_map = {
            'encoder._transformer_encoder.self_attns.{layer}.{c}_dense.weight':
                'encoder.layers.{layer}.self_attn.{c}_proj.weight',
            'encoder._transformer_encoder.self_attns.{layer}.{c}_dense.bias':
                'encoder.layers.{layer}.self_attn.{c}_proj.bias'
        }

        layer_map = {
            'encoder._transformer_encoder.output_layer_norm.{layer}.weight':
                'encoder.layers.{layer}.final_layer_norm.weight',
            'encoder._transformer_encoder.output_layer_norm.{layer}.bias':
                'encoder.layers.{layer}.final_layer_norm.bias',
            'encoder._transformer_encoder.poswise_networks.{layer}._layers.0.'
            'weight': 'encoder.layers.{layer}.fc1.weight',
            'encoder._transformer_encoder.poswise_networks.{layer}._layers.0.'
            'bias': 'encoder.layers.{layer}.fc1.bias',
            'encoder._transformer_encoder.poswise_networks.{layer}._layers.3.'
            'weight': 'encoder.layers.{layer}.fc2.weight',
            'encoder._transformer_encoder.poswise_networks.{layer}._layers.3.'
            'bias': 'encoder.layers.{layer}.fc2.bias',
            'encoder._transformer_encoder.poswise_layer_norm.{layer}.weight':
                'encoder.layers.{layer}.self_attn_layer_norm.weight',
            'encoder._transformer_encoder.poswise_layer_norm.{layer}.bias':
                'encoder.layers.{layer}.self_attn_layer_norm.bias'
        }

        for temp_ours, temp_ckpt in no_cond_map.items():
            self.state_dict()[temp_ours].copy_(ckpt_state_dict[temp_ckpt])

        for layer in range(12):
            for temp_ours, temp_ckpt in layer_map.items():
                self.state_dict()[temp_ours.format(layer=layer)].copy_(
                    ckpt_state_dict[temp_ckpt.format(layer=layer)])

            for c in ['Q', 'K', 'V', 'O']:
                for temp_ours, temp_ckpt in layer_c_map.items():
                    self.state_dict()[temp_ours.format(layer=layer, c=c)].copy_(
                        ckpt_state_dict[temp_ckpt.format(layer=layer, c=c)])


    def _transform_config(cls, pretrained_model_name: str,
                          cache_dir: str):
        pass


if __name__ == '__main__':
    bart_mixin = PretrainedBARTMixin()
    bart_mixin.init_pretrained_weights(
        pretrained_model_name='bart.large', cache_dir='.')