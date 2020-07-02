import torch
from torch import nn

from model.bart_encoder_decoder import BART

from fairseq.models.bart.model import BARTModel
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

src_tokens = torch.tensor([input_ids])
src_lengths = torch.tensor([len(input_ids)])
decoder_input = torch.tensor([input_ids])

# print(bart)

# print(len(list(bart.named_parameters())))
#
# total_numel = 0
# for name, param in bart.named_parameters():
#     print(name, param.shape)
#     total_numel += param.numel()
# print(total_numel)

try:
    bart(src_tokens=src_tokens, src_lengths=src_lengths,
         decoder_input=decoder_input[:, -1], labels=decoder_input[:, 1:])
except:
    fs_bart.model(src_tokens=src_tokens, src_lengths=src_lengths, prev_output_tokens=decoder_input)
