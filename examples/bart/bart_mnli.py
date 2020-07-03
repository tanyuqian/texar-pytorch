from tqdm import tqdm

import torch

from model.bart_encoder_decoder import BART

BATCH_SIZE = 8


def main():
    bart = BART(pretrained_model_name='bart.large.mnli').to('cuda')
    bart.eval()

    batches = [[]]
    for line in open('glue_data/MNLI/dev_matched.tsv').readlines():
        line = line.strip().split('\t')
        sent1, sent2, target = line[8], line[9], line[-1]

        batches[-1].append([sent1, sent2, target])
        if len(batches[-1]) == BATCH_SIZE:
            batches.append([])

    label_map = {0: 'contradiction', 1: 'neutral', 2: 'entailment'}
    n_correct, n_sample = 0, 0
    for batch in tqdm(batches, desc='Testing'):
        tokens = [bart.encode(sent1, sent2) for sent1, sent2, target in batch]
        tokens, lengths = bart.make_batch(tokens)

        logits = bart.predict(head='mnli', tokens=tokens, lengths=lengths)
        preds = torch.argmax(logits, dim=-1).tolist()

        n_correct += sum([1 for i in range(len(batch))
                          if label_map[preds[i]] == batch[i][-1]])
        n_sample += len(batch)

        print('| Accuracy: ', float(n_correct) / float(n_sample))
    #
    # with open('glue_data/MNLI/dev_matched.tsv') as fin:
    #     fin.readline()
    #     for index, line in enumerate(fin):
    #         line = line.strip().split('\t')
    #         sent1, sent2, target = line[8], line[9], line[-1]
    #
    #         tokens = bart.encode(sent1, sent2)
    #         tokens, lengths = \
    #             torch.tensor([tokens]).to('cuda'),\
    #             torch.tensor([len(tokens)]).to('cuda')
    #         pred = bart.predict('mnli', tokens=tokens, lengths=lengths).argmax().item()
    #
    #         pred_label = label_map[pred]
    #         ncorrect += int(pred_label == target)
    #         nsamples += 1
    #         print('| Accuracy: ', float(ncorrect)/float(nsamples))
    # # Expected output: 0.9010


if __name__ == '__main__':
    main()
