"""
This is a simple implementation of a tokenizer for English sentences.

First, it builds a vocabulary from the corpus.
When you input a sentence (or a sequence of words), it encodes it into a list of token IDs.
When a sentence is required, it decodes a list of token IDs back into a sentence.

The tokenizer also handles special tokens like:
    _ <pad> - padding (to make all sequences the same length)
    _ <sos> - start of sentence
    _ <eos> - end of sentence
    _ <unk> - unknown token (for words not in the vocabulary)
    
The tokenizer is responsible for converting raw text into a format that can be fed into a machine learning model (numeric vector), and vice versa.
"""
class Tokenizer:
    def __init__(self):
        self.stoi = {}
        self.itos = {}
        self.vocab = []
        self.special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"]

    def build_vocab(self, sentences):
        words = []
        
        for sentence in sentences:
            tokens = sentence.lower().split()
            words.extend(tokens)

        unique_words = sorted(list(set(words)))
        self.vocab = (self.special_tokens + unique_words)
        self.stoi = {s: i for i, s in enumerate(self.vocab)}
        self.itos = {i: s for s, i in self.stoi.items()}

    def encode(self, sentence):
        tokens = sentence.lower().split()
        encoded = [self.stoi["<sos>"]]

        for token in tokens:
            # If the token is in the vocab, use its index
            if token in self.stoi: encoded.append(self.stoi[token])
            # Otherwise, use the index for <unk>
            else: encoded.append(self.stoi["<unk>"])

        encoded.append(self.stoi["<eos>"])
        return encoded


    def decode(self, token_ids):
        words = []

        for idx in token_ids:
            word = self.itos[idx]
            if word in ["<sos>", "<eos>", "<pad>"]: continue
            words.append(word)

        return " ".join(words)