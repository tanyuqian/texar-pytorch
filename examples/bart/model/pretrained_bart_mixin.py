from texar.torch.modules.pretrained.pretrained_base import PretrainedMixin


class PretrainedBARTMixin(PretrainedMixin):
    _MODEL_NAME = "BART"
    _MODEL2URL = {
        'bart.large': 'https://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz'
    }

    def __init__(self):
        pass

    def init_pretrained_weights(self, pretrained_model_name, cache_dir):
        self.pretrained_model_dir = self.download_checkpoint(
            pretrained_model_name, cache_dir)

    def _init_from_checkpoint(self, pretrained_model_name: str,
                              cache_dir: str, **kwargs):
        pass

    def _transform_config(cls, pretrained_model_name: str,
                          cache_dir: str):
        pass


if __name__ == '__main__':
    bart_mixin = PretrainedBARTMixin()
    bart_mixin.init_pretrained_weights(pretrained_model_name='bart.large', cache_dir='.')