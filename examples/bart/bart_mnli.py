import torch

from model.bart_encoder_decoder import BART


def main():
    bart = BART(pretrained_model_name='bart.large.mnli').to('cuda')
    bart.eval()

    label_map = {0: 'contradiction', 1: 'neutral', 2: 'entailment'}
    ncorrect, nsamples = 0, 0

    with open('glue_data/MNLI/dev_matched.tsv') as fin:
        fin.readline()
        for index, line in enumerate(fin):
            line = line.strip().split('\t')
            sent1, sent2, target = line[8], line[9], line[-1]

            tokens = bart.encode(sent1, sent2)
            tokens, lengths = \
                torch.tensor([tokens]).to('cuda'),\
                torch.tensor([len(tokens)]).to('cuda')
            pred = bart.predict('mnli', tokens=tokens, lengths=lengths).argmax().item()

            pred_label = label_map[pred]
            ncorrect += int(pred_label == target)
            nsamples += 1
            print('| Accuracy: ', float(ncorrect)/float(nsamples))
    # Expected output: 0.9010


if __name__ == '__main__':
    main()
