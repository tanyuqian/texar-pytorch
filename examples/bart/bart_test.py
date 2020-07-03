import torch
from torch import nn

from model.bart_encoder_decoder import BART

from fairseq.models.bart import BARTHubInterface, BARTModel
from fairseq.models.transformer import TransformerEncoder


example = '''Texar-PyTorch is a toolkit aiming to support a broad set of machine
learning, especially natural language processing and text generation tasks.
Texar provides a library of easy-to-use ML modules and functionalities for
composing whatever models and algorithms. The tool is designed for both
researchers and practitioners for fast prototyping and experimentation.'''

bart = BART(pretrained_model_name='bart.large.cnn')
bart.eval()
# input_ids = bart.encode(
#     'BART is a sequence model.', 'BART is not sequence to sequence.')
input_ids = bart.encode(example)


# for name, param in bart.named_parameters():
#     print(name, param.shape)
# exit()

fs_bart = torch.hub.load('pytorch/fairseq', 'bart.large.cnn')
fs_bart.eval()
fs_input_ids = fs_bart.encode(example).tolist()
# fs_input_ids = fs_bart.encode(
#     'BART is a sequence model.', 'BART is not sequence to sequence.').tolist()
# #
print(input_ids)
print(fs_input_ids)
assert input_ids == fs_input_ids

src_tokens = torch.tensor([input_ids])
src_lengths = torch.tensor([len(input_ids)])
tgt_tokens = torch.tensor([[0]])

# print(bart.extract_features(tokens=tokens, lengths=lengths))
# print(fs_bart.extract_features(tokens=tokens))

print(bart(src_tokens=src_tokens, src_lengths=src_lengths, decoder_input=tgt_tokens))

print(fs_bart.model(src_tokens=src_tokens, src_lengths=src_lengths, prev_output_tokens=tgt_tokens)[0])

# print(fs_bart.predict(head='mnli', tokens=tokens))
