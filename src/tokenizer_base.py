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

import re
import unicodedata
import json
from abc import ABC, abstractmethod

class TokenizerBase(ABC):
    def __init__(self):
        self.vocab = []
        self.stoi = {}
        self.itos = {}
        self.special_tokens = ["<pad>", "<sos>", "<eos>", "<unk>"]

    def sos_id(self):
        return self.stoi["<sos>"]
    
    def eos_id(self):
        return self.stoi["<eos>"]

    def tokenize(self, text):
        text = unicodedata.normalize("NFKC", text)
        text = text.lower().strip()
        pattern = r"""
            \d+\.\d+           | # decimal numbers
            \w+(?:[-']\w+)*    | # words with - or '
            [^\w\s]              # punctuation
        """
        tokens = re.findall(pattern, text, re.VERBOSE)

        return tokens

    def save_vocab(self, path):
        data = {"vocab": self.vocab, "stoi": self.stoi}
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        print(f"Vocabulary saved to {path}")
        
    def load_vocab(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        self.vocab = data["vocab"]
        self.stoi = {k: int(v) for k, v in data["stoi"].items()}
        self.itos = {i: s for s, i in self.stoi.items()}
        print(f"Vocabulary loaded from {path}")

    @abstractmethod
    def build_vocab(self, sentences): pass

    @abstractmethod
    def encode(self, text): pass

    @abstractmethod
    def decode(self, token_ids): pass