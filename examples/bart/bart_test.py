import torch

from model.bart_tokenizer import BARTTokenizer

example = '''
Texar-PyTorch is a toolkit aiming to support a broad set of machine learning, 
especially natural language processing and text generation tasks. Texar provides
 a library of easy-to-use ML modules and functionalities for composing whatever 
 models and algorithms. The tool is designed for both researchers and 
 practitioners for fast prototyping and experimentation.'''

bart = torch.hub.load('pytorch/fairseq', 'bart.large')

tk = BARTTokenizer()

print(tk.encode(example))
print(bart.encode(example))