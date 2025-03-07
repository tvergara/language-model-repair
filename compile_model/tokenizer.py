import torch

class Tokenizer:
    def __init__(self, encoding_map,decoding_map,max_length):
        self.vocab = encoding_map
        self.decoder_vocab = {v:k for k,v in decoding_map.items()}
        self.max_length = max_length
        if sum([type(n)==int for n in self.vocab.keys()]):
            self.vocab = {str(k):v for k,v in self.vocab.items()}

    def tokenize(self, text, return_tensor='pt', add_special_tokens=True):
        tokens = []
        for word in text.strip().split():

            if word in self.vocab:
                tokens.append(self.vocab[word])
        special_tokens = [self.vocab.get('bos')] if add_special_tokens else []
        tokens = [] + tokens[-self.max_length+1:]
        if return_tensor == 'pt':
            return torch.tensor(tokens)
        return tokens

    def encode(self, text):
        return self.vocab[text]

    def __call__(self, text, return_tensor='pt', add_special_tokens=True):
        return self.tokenize(text, return_tensor, add_special_tokens)

    def decode(self,output):
        texts = [[] * output.size(0)]
        output = output.cpu().tolist()

        for n in range(len(output)):
            texts[n] = [str(self.decoder_vocab[x]) for x in output[n]]
        return ['bos'+t[1:] for t in texts]
