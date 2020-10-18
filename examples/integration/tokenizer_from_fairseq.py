from typing import Optional, List, Any

from texar.torch.data.tokenizers.tokenizer_base import TokenizerBase


class TokenizerFromFairseq(TokenizerBase):
    def __init__(self, fs_tokenizer):
        TokenizerBase.__init__(self, hparams=None)

        self._fs_tokenizer = fs_tokenizer

    def map_text_to_id(self, text):
        return [int(t) for t in self._fs_tokenizer.encode(text).split()]

    def map_id_to_text(self, token_ids,
                       skip_special_tokens=False,
                       clean_up_tokenization_spaces=True):
        token_ids_str = ' '.join([str(id) for id in token_ids])
        return self._fs_tokenizer.decode(token_ids_str)

