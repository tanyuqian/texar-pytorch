import pickle

from texar.torch.data.data_utils import maybe_download
from texar.torch.data.tokenizers.gpt2_tokenizer import GPT2Tokenizer


class BARTTokenizer:
    def __init__(self):
        mapping_path = maybe_download(
            urls='https://drive.google.com/file/d/'
                 '1bXfw761TkOfA45IxQUSYSmTKp4QEvQWH/view?usp=sharing',
            path='/tmp', filenames='gpt2_to_bart.pickle')

        self._gpt2_tokenizer = GPT2Tokenizer(pretrained_model_name='gpt2-small')

        self._bos_id = 0
        self._eos_id = 2
        self._pad_id = 1

        self._gpt2_to_bart = pickle.load(open(mapping_path, 'rb'))
        self._bart_to_gpt2 = {
            value: key for key, value in self._gpt2_to_bart.items()}

    def encode(self, sentence, *additional_sentences):
        gpt2_ids, len = self._gpt2_tokenizer.encode_text(
            text=sentence, max_seq_length=1024, append_eos_token=False)

        ids = [self.bos_id] + \
              [self._gpt2_to_bart[str(t)] for t in gpt2_ids[1:len]] + \
              [self.eos_id]

        for sent in additional_sentences:
            gpt2_ids, len = self._gpt2_tokenizer.encode_text(
                text=sent, max_seq_length=1024, append_eos_token=False)
            ids.extend([self._gpt2_to_bart[str(t)] for t in gpt2_ids[1:len]] +
                       [self.eos_id])

        return ids

    def decode(self, token_ids):
        gpt2_ids = []
        if token_ids[0] == self._bos_id:
            token_ids = token_ids[1:]
        while len(token_ids) > 0 and token_ids[-1] in [
            self.bos_id, self.eos_id, self.pad_id]:
            token_ids = token_ids[:-1]
        token_ids = token_ids + [self.eos_id]

        sentences = []
        for t in token_ids:
            if t != self._eos_id:
                gpt2_ids.append(int(self._bart_to_gpt2[t]))
            else:
                sentences.append(self._gpt2_tokenizer.map_id_to_text(
                    token_ids=gpt2_ids))
                gpt2_ids = []

        return sentences

    @property
    def pad_id(self):
        return self._pad_id

    @property
    def bos_id(self):
        return self._bos_id

    @property
    def eos_id(self):
        return self._eos_id

    @property
    def vocab_size(self):
        return max(self._gpt2_to_bart.values()) + 1