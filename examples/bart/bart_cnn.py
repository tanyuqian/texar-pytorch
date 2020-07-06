from tqdm import trange

from model.bart_encoder_decoder import BART

BATCH_SIZE = 1


def main():
    bart = BART(pretrained_model_name='bart.large.cnn').to('cuda')
    bart.eval()

    test_src_file = open('cnn_dm/test.source')
    test_hypo_file = open('cnndm.hypo', 'w')

    src_sents = [line.strip() for line in test_src_file.readlines()]

    for i in trange(0, len(src_sents), BATCH_SIZE, desc='Testing CNN/DM'):
        hypos = bart.sample(
            src_sents[i: i + BATCH_SIZE],
            beam_width=2,
            length_penalty=2.,
            max_decoding_length=140)

        for hypo in hypos:
            print(hypo, file=test_hypo_file)
        test_hypo_file.flush()


if __name__ == '__main__':
    main()
