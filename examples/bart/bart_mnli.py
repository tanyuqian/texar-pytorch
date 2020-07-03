from tqdm import tqdm

import torch

from model.bart_encoder_decoder import BART

BATCH_SIZE = 8


def main():
    fs_bart = torch.hub.load('pytorch/fairseq', 'bart.large.cnn').to('cuda')
    fs_bart.eval()

    bart = BART(pretrained_model_name='bart.large.mnli').to('cuda')
    bart.eval()

    batches = [[]]
    for line in open('glue_data/MNLI/dev_matched.tsv').readlines():
        line = line.strip().split('\t')
        sent1, sent2, target = line[8], line[9], line[-1]

        batches[-1].append([sent1, sent2, target])
        if len(batches[-1]) == BATCH_SIZE:
            batches.append([])

    if batches[-1] == []:
        batches = batches[:-1]

    label_map = {0: 'contradiction', 1: 'neutral', 2: 'entailment'}
    n_correct, n_sample = 0, 0
    for batch in tqdm(batches, desc='Testing'):
        tokens = [bart.encode(sent1, sent2) for sent1, sent2, target in batch]
        tokens, lengths = bart.make_batch(tokens)

        logits = bart.predict(head='mnli', tokens=tokens, lengths=lengths)
        preds = torch.argmax(logits, dim=-1).tolist()

        for sent1, sent2, target in batch:
            fs_tokens = fs_bart.encode(sent1, sent2).tolist()
            if bart.encode(sent1, sent2) != fs_tokens:
                print(sent1)
                print(sent2)
                print(bart.encode(sent1, sent2))
                print(fs_tokens)

        n_correct += sum([1 for i in range(len(batch))
                          if label_map[preds[i]] == batch[i][-1]])
        n_sample += len(batch)

        print('| Accuracy: ', float(n_correct) / float(n_sample))


if __name__ == '__main__':
    main()
