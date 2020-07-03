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
tgt_tokens = [0]

preds = bart.generate(
    src_tokens=src_tokens,
    src_lengths=src_lengths,
    beam_width=4,
    length_penalty=2.,
    max_decoding_length=140)

print(preds)

# print(bart.extract_features(tokens=tokens, lengths=lengths))
# print(fs_bart.extract_features(tokens=tokens))

# for t in range(1000):
#     logits_ours = bart(
#         src_tokens=src_tokens, src_lengths=src_lengths,
#         decoder_input=torch.tensor([tgt_tokens])).logits[:, -1]
#
#     logits_fs = fs_bart.model(
#         src_tokens=src_tokens, src_lengths=src_lengths,
#         prev_output_tokens=torch.tensor([tgt_tokens]))[0][:, -1]
#
#     id_ours = torch.argmax(logits_ours.view(-1)).item()
#     id_fs = torch.argmax(logits_fs.view(-1)).item()
#
#     assert id_ours == id_fs
#
#     print(logits_ours.shape, logits_fs.shape)
#     print('id:', id_ours, id_fs)
#
#     tgt_tokens.append(id_ours)
#
#     print(f'Step {t}: {tgt_tokens}')
#     print(bart.decode(tgt_tokens))
#
#     if id_ours == 2:
#         break
#
# print(fs_bart.sample([example], beam=1))
