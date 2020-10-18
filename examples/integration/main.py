from fairseq.data.encoders.gpt2_bpe import GPT2BPE

from tokenizer_from_fairseq import TokenizerFromFairseq


sentence = 'You are a good man.'

fs_tokenizer = GPT2BPE(args=None)

fs_ids = fs_tokenizer.encode(sentence)
print('Fairseq IDS:', fs_ids)
print('Fairseq Decode:', fs_tokenizer.decode(fs_ids))

tok = TokenizerFromFairseq(fs_tokenizer=fs_tokenizer)

tx_ids = tok.map_text_to_id(sentence)
print('Texar IDS:', tx_ids)
print('Texar Decode:', tok.map_id_to_text(tx_ids))


