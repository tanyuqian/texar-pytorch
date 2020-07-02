import torch
from torch import nn

from model.bart_encoder_decoder import BART

from fairseq.models.bart import BARTHubInterface
from fairseq.models.transformer import TransformerEncoder


example = '''Texar-PyTorch is a toolkit aiming to support a broad set of machine
learning, especially natural language processing and text generation tasks.
Texar provides a library of easy-to-use ML modules and functionalities for
composing whatever models and algorithms. The tool is designed for both
researchers and practitioners for fast prototyping and experimentation.'''
#
# bart = torch.hub.load('pytorch/fairseq', 'bart.large')
#
# tk = BARTTokenizer()
#
# if tk.encode(example) == bart.encode(example).tolist():
#     print('Tokenizer Check Passed.')
# else:
#     raise ValueError('Tokenizer Check Failed!!!')

bart = BART()
bart.eval()
input_ids = bart.encode(example)

fs_bart = torch.hub.load('pytorch/fairseq', 'bart.large')
fs_bart.eval()
fs_input_ids = fs_bart.encode(example).tolist()

assert input_ids == fs_input_ids

tokens = torch.tensor([input_ids])

print(bart.extract_features(tokens=tokens))
print('=' * 50)
print(fs_bart.extract_features(tokens=tokens))