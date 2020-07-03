import os

import torch

from texar.torch.modules.pretrained.pretrained_base import PretrainedMixin


class PretrainedBARTMixin(PretrainedMixin):
    _MODEL_NAME = "BART"
    _MODEL2URL = {
        'bart.large': 'https://drive.google.com/file/d/1IvBWpjjfcEK7LBIsZbw37rUiLHqnLW77/view?usp=sharing',
        'bart.large.cnn': 'https://drive.google.com/file/d/1HWL_4wmoUC9JooT1r4KCOlm2Awcpcjfj/view?usp=sharing',
        'bart.large.mnli': 'https://drive.google.com/file/d/1lo2PAmfmCsRT0g7SWa2_J8YpafX1JBcS/view?usp=sharing'
    }
    _MODEL2CKPT = {
        'bart.large': 'model.pt',
        'bart.large.cnn': 'model.pt',
        'bart.large.mnli': 'model.pt'
    }

    def __init__(self):
        PretrainedMixin.__init__(self)

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
            'decoder._pos_embedder.weight': 'decoder.embed_positions.weight',
            'decoder._layernorm_embedding.weight':
                'decoder.layernorm_embedding.weight',
            'decoder._layernorm_embedding.bias':
                'decoder.layernorm_embedding.bias',
        }
        # if 'mnli' in pretrained_model_name:
        #     no_cond_map['head_mnli.0.weight'] = \
        #         'classification_heads.mnli.dense.weight'
        #     no_cond_map['head_mnli.0.bias'] = \
        #         'classification_heads.mnli.dense.bias'
        #     no_cond_map['head_mnli.2.weight'] = \
        #         'classification_heads.mnli.out_proj.weight'
        #     no_cond_map['head_mnli.2.bias'] = \
        #         'classification_heads.mnli.out_proj.bias'

        layer_c_map = {
            'encoder._transformer_encoder.self_attns.{layer}.{c}_dense.weight':
                'encoder.layers.{layer}.self_attn.{c}_proj.weight',
            'encoder._transformer_encoder.self_attns.{layer}.{c}_dense.bias':
                'encoder.layers.{layer}.self_attn.{c}_proj.bias',
            'decoder._transformer_decoder.self_attns.{layer}.{c}_dense.weight':
                'decoder.layers.{layer}.self_attn.{c}_proj.weight',
            'decoder._transformer_decoder.self_attns.{layer}.{c}_dense.bias':
                'decoder.layers.{layer}.self_attn.{c}_proj.bias',
            'decoder._transformer_decoder.enc_dec_attns.{layer}.{c}_dense.'
            'weight': 'decoder.layers.{layer}.encoder_attn.{c}_proj.weight',
            'decoder._transformer_decoder.enc_dec_attns.{layer}.{c}_dense.'
            'bias': 'decoder.layers.{layer}.encoder_attn.{c}_proj.bias',
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
                'encoder.layers.{layer}.self_attn_layer_norm.bias',
            'decoder._transformer_decoder.self_attn_layer_norm.{layer}.weight':
                'decoder.layers.{layer}.self_attn_layer_norm.weight',
            'decoder._transformer_decoder.self_attn_layer_norm.{layer}.bias':
                'decoder.layers.{layer}.self_attn_layer_norm.bias',
            'decoder._transformer_decoder.end_dec_attn_layer_norm.{layer}.'
            'weight': 'decoder.layers.{layer}.encoder_attn_layer_norm.weight',
            'decoder._transformer_decoder.end_dec_attn_layer_norm.{layer}.'
            'bias': 'decoder.layers.{layer}.encoder_attn_layer_norm.bias',
            'decoder._transformer_decoder.poswise_networks.{layer}._layers.0.'
            'weight': 'decoder.layers.{layer}.fc1.weight',
            'decoder._transformer_decoder.poswise_networks.{layer}._layers.0.'
            'bias': 'decoder.layers.{layer}.fc1.bias',
            'decoder._transformer_decoder.poswise_networks.{layer}._layers.3.'
            'weight': 'decoder.layers.{layer}.fc2.weight',
            'decoder._transformer_decoder.poswise_networks.{layer}._layers.3.'
            'bias': 'decoder.layers.{layer}.fc2.bias',
            'decoder._transformer_decoder.poswise_layer_norm.{layer}.weight':
                'decoder.layers.{layer}.final_layer_norm.weight',
            'decoder._transformer_decoder.poswise_layer_norm.{layer}.bias':
                'decoder.layers.{layer}.final_layer_norm.bias',
        }

        for temp_ours, temp_ckpt in no_cond_map.items():
            ours_name = temp_ours
            ckpt_name = temp_ckpt
            self.state_dict()[ours_name].copy_(ckpt_state_dict[ckpt_name])

        for layer in range(12):
            for temp_ours, temp_ckpt in layer_map.items():
                ours_name = temp_ours.format(layer=layer)
                ckpt_name = temp_ckpt.format(layer=layer)
                self.state_dict()[ours_name].copy_(ckpt_state_dict[ckpt_name])

            for c in ['Q', 'K', 'V', 'O']:
                for temp_ours, temp_ckpt in layer_c_map.items():
                    ours_name = temp_ours.format(layer=layer, c=c)
                    ckpt_name = temp_ckpt.format(
                        layer=layer, c=c.lower() if c != 'O' else 'out')
                    self.state_dict()[ours_name].copy_(
                        ckpt_state_dict[ckpt_name])

    def _transform_config(cls, pretrained_model_name: str,
                          cache_dir: str):
        pass


if __name__ == '__main__':
    bart_mixin = PretrainedBARTMixin()
    bart_mixin.init_pretrained_weights(
        pretrained_model_name='bart.large', cache_dir='.')