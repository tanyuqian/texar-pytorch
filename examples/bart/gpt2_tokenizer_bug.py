from texar.torch.data.tokenizers import GPT2Tokenizer

tokenizer = GPT2Tokenizer(pretrained_model_name='gpt2-small')

example = 'BART is a seq2seq model.'

ids = tokenizer.map_text_to_id(text=example)

print('original text:\n', example)
print('text -> ids -> text:\n', tokenizer.map_id_to_text(ids))
