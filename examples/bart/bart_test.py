import torch
from torch import nn

from model.bart_encoder_decoder import BART


# example = '''Texar-PyTorch is a toolkit aiming to support a broad set of machine
# learning, especially natural language processing and text generation tasks.
# Texar provides a library of easy-to-use ML modules and functionalities for
# composing whatever models and algorithms. The tool is designed for both
# researchers and practitioners for fast prototyping and experimentation.'''
#
# bart = torch.hub.load('pytorch/fairseq', 'bart.large')
#
# tk = BARTTokenizer()
#
# if tk.encode(example) == bart.encode(example).tolist():
#     print('Tokenizer Check Passed.')
# else:
#     raise ValueError('Tokenizer Check Failed!!!')

bart = BART()._decoder

print(bart)

# print(len(list(bart.named_parameters())))
#
# total_numel = 0
# for name, param in bart.named_parameters():
#     print(name, param.shape)
#     total_numel += param.numel()
# print(total_numel)
