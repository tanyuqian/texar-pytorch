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

# print(bart.extract_features(tokens=tokens, lengths=lengths))
# print(fs_bart.extract_features(tokens=tokens))

for t in range(10):
    logits_ours = bart(
        src_tokens=src_tokens, src_lengths=src_lengths,
        decoder_input=torch.tensor([tgt_tokens]))

    logits_fs = fs_bart.model(
        src_tokens=src_tokens, src_lengths=src_lengths,
        prev_output_tokens=torch.tensor([tgt_tokens]))[0]

    print(f'Step {t}')
    print(logits_ours)
    print(logits_fs)

    id = torch.argmax(logits_ours[0]).item()
    tgt_tokens.append(id)

# print(fs_bart.predict(head='mnli', tokens=tokens))
