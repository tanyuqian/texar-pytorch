import torch
from torch import nn

from model.bart_encoder_decoder import BART

from fairseq.models.bart import BARTHubInterface, BARTModel
from fairseq.models.transformer import TransformerEncoder


example = '''The Palestinian Authority officially became the 123rd member of the International Criminal Court on Wednesday, a step that gives the court jurisdiction over alleged crimes in Palestinian territories. The formal accession was marked with a ceremony at The Hague, in the Netherlands, where the court is based. The Palestinians signed the ICC's founding Rome Statute in January, when they also accepted its jurisdiction over alleged crimes committed "in the occupied Palestinian territory, including East Jerusalem, since June 13, 2014." Later that month, the ICC opened a preliminary examination into the situation in Palestinian territories, paving the way for possible war crimes investigations against Israelis. As members of the court, Palestinians may be subject to counter-charges as well. Israel and the United States, neither of which is an ICC member, opposed the Palestinians' efforts to join the body. But Palestinian Foreign Minister Riad al-Malki, speaking at Wednesday's ceremony, said it was a move toward greater justice. "As Palestine formally becomes a State Party to the Rome Statute today, the world is also a step closer to ending a long era of impunity and injustice," he said, according to an ICC news release. "Indeed, today brings us closer to our shared goals of justice and peace." Judge Kuniko Ozaki, a vice president of the ICC, said acceding to the treaty was just the first step for the Palestinians. "As the Rome Statute today enters into force for the State of Palestine, Palestine acquires all the rights as well as responsibilities that come with being a State Party to the Statute. These are substantive commitments, which cannot be taken lightly," she said. Rights group Human Rights Watch welcomed the development. "Governments seeking to penalize Palestine for joining the ICC should immediately end their pressure, and countries that support universal acceptance of the court's treaty should speak out to welcome its membership," said Balkees Jarrah, international justice counsel for the group. "What's objectionable is the attempts to undermine international justice, not Palestine's decision to join a treaty to which over 100 countries around the world are members." In January, when the preliminary ICC examination was opened, Israeli Prime Minister Benjamin Netanyahu described it as an outrage, saying the court was overstepping its boundaries. The United States also said it "strongly" disagreed with the court's decision. "As we have said repeatedly, we do not believe that Palestine is a state and therefore we do not believe that it is eligible to join the ICC," the State Department said in a statement. It urged the warring sides to resolve their differences through direct negotiations. "We will continue to oppose actions against Israel at the ICC as counterproductive to the cause of peace," it said. But the ICC begs to differ with the definition of a state for its purposes and refers to the territories as "Palestine." While a preliminary examination is not a formal investigation, it allows the court to review evidence and determine whether to investigate suspects on both sides. Prosecutor Fatou Bensouda said her office would "conduct its analysis in full independence and impartiality." The war between Israel and Hamas militants in Gaza last summer left more than 2,000 people dead. The inquiry will include alleged war crimes committed since June. The International Criminal Court was set up in 2002 to prosecute genocide, crimes against humanity and war crimes. CNN's Vasco Cotovio, Kareem Khadder and Faith Karimi contributed to this report.'''

bart = BART(pretrained_model_name='bart.large.cnn').to('cuda')
bart.eval()
# input_ids = bart.encode(
#     'BART is a sequence model.', 'BART is not sequence to sequence.')
input_ids = bart.encode(example)[:1024]


# for name, param in bart.named_parameters():
#     print(name, param.shape)
# exit()

# fs_bart = torch.hub.load('pytorch/fairseq', 'bart.large.cnn').to('cuda')
# fs_bart.eval()
# fs_input_ids = fs_bart.encode(example).tolist()
# fs_input_ids = fs_bart.encode(
#     'BART is a sequence model.', 'BART is not sequence to sequence.').tolist()
# #
# assert input_ids == fs_input_ids

src_tokens = torch.tensor([input_ids]).to('cuda')
src_lengths = torch.tensor([len(input_ids)]).to('cuda')
tgt_tokens = [0]

# print(bart.sample([example, example]))

# preds = bart.generate(
#     src_tokens=src_tokens,
#     src_lengths=src_lengths,
#     beam_width=4,
#     length_penalty=2.,
#     max_decoding_length=140)
#
# print(preds)

# print(bart.extract_features(tokens=tokens, lengths=lengths))
# print(fs_bart.extract_features(tokens=tokens))

for t in range(1000):
    logits_ours = bart(
        src_tokens=src_tokens, src_lengths=src_lengths,
        decoder_input=torch.tensor([tgt_tokens]).to('cuda')).logits[:, -1]

    # logits_fs = fs_bart.model(
    #     src_tokens=src_tokens, src_lengths=src_lengths,
    #     prev_output_tokens=torch.tensor([tgt_tokens]).to('cuda'))[0][:, -1]

    id_ours = torch.argmax(logits_ours.view(-1)).item()
    # id_fs = torch.argmax(logits_fs.view(-1)).item()

    # assert id_ours == id_fs

    # print(logits_ours.shape, logits_fs.shape)
    # print('id:', id_ours, id_fs)

    tgt_tokens.append(id_ours)

    print(f'Step {t}: {tgt_tokens}')
    print(bart.decode(tgt_tokens))

    if id_ours == 2:
        break

# print(fs_bart.sample([example], beam=1))
