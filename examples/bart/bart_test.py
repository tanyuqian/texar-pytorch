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

bart = BART(pretrained_model_name='bart.large.mnli')
bart.eval()
# input_ids = bart.encode(
#     'BART is a sequence model.', 'BART is not sequence to sequence.')
input_ids = bart.encode(example)


# for name, param in bart.named_parameters():
#     print(name, param.shape)
# exit()

fs_bart = torch.hub.load('pytorch/fairseq', 'bart.large.cnn')
fs_bart.eval()
# fs_input_ids = fs_bart.encode(
#     'BART is a sequence model.', 'BART is not sequence to sequence.').tolist()
# #
# print(input_ids)
# print(fs_input_ids)
# assert input_ids == fs_input_ids

tokens = torch.tensor([input_ids])
lengths = torch.tensor([len(input_ids)])

# print(bart.extract_features(tokens=tokens, lengths=lengths))
# print(fs_bart.extract_features(tokens=tokens))

# sample_id = bart.generate(
#     src_tokens=tokens, src_lengths=lengths)['sample_id'][:, :, 0].view(-1).tolist()
# print(sample_id)
# print(bart.decode(sample_id))

print(fs_bart.sample([example], beam=4, lenpen=2.0, max_len_b=140))

# print(fs_bart.predict(head='mnli', tokens=tokens))
