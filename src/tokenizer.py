from .tokenizer_base import TokenizerBase

# ================================================== SIMPLE TOKENIZER ==================================================
# ======================================================================================================================

"""
    The rule is very simple: each word is a token. 
"""

class SimpleTokenizer(TokenizerBase):
    def __init__(self):
        super().__init__()

    def build_vocab(self, sentences, ctype, ttype, lang="en"):
        words = []
        
        for sentence in sentences:
            tokens = self.tokenize(sentence)
            words.extend(tokens)

        unique_words = sorted(list(set(words)))
        self.vocab = (self.special_tokens + unique_words)
        self.stoi = {s: i for i, s in enumerate(self.vocab)}
        self.itos = {i: s for s, i in self.stoi.items()}

    def encode(self, text):
        tokens = self.tokenize(text)
        encoded = [self.stoi["<sos>"]]

        for token in tokens:
            encoded.append(self.stoi.get(token, self.stoi["<unk>"]))

        encoded.append(self.stoi["<eos>"])
        return encoded


    def decode(self, token_ids):
        words = []

        for idx in token_ids:
            token = self.itos[idx]
            if token in ["<sos>", "<eos>", "<pad>"]: continue
            words.append(token)

        return " ".join(words)
    

# ================================================== BPE TOKENIZER ==================================================
# ===================================================================================================================

"""
What's new: sub-word spliting funtionality using Byte Pair Encoding (BPE) technique (the underlying tokenizing method of GPT).
BPE:
    1. Build initial vocab (by spliting words into multiple single character)
    2. Determine the frequency of occurrence of each pair
    3. Merge each pair into a single token, then return to Step 2.

Result: Frequency dictionary of each pairs appearing in the corpus. 
"""

from collections import Counter

class BPETokenizer(TokenizerBase):
    def __init__(self):
        super().__init__()
        self.bpe_vocab = {}
        self.merges = []

    def initial_vocab(self, sentences):
        words = []
        for sentence in sentences:
            words.extend(self.tokenize(sentence))

        word_freqs = Counter(words)

        bpe_vocab = {}
        for word, freq in word_freqs.items():
            chars = " ".join(list(word))
            chars += " </w>"
            bpe_vocab[chars] = freq

        self.bpe_vocab = bpe_vocab
        
    def get_stats(self):
        pairs = Counter()     
        for word, freq in self.bpe_vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pairs[pair] += freq

        return pairs
    
    def merge_pair(self, pair):
        new_vocab = {}
        bigram = " ".join(pair)
        replacement = "".join(pair)

        for word in self.bpe_vocab:
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = self.bpe_vocab[word]

        self.bpe_vocab = new_vocab
        self.merges.append(pair)
        
    def final_vocab(self):
        tokens = set()

        for word in self.bpe_vocab:
            pieces = word.split()
            tokens.update(pieces)

        self.vocab = (self.special_tokens + sorted(list(tokens)))
        self.stoi = {s: i for i, s in enumerate(self.vocab)}
        self.itos = {i: s for s, i in self.stoi.items()}
        
    def build_vocab(self, sentences, ctype="initial", ttype="BPE", lang="en", num_merges=500):
        self.initial_vocab(sentences)
        
        for i in range(num_merges):
            pairs = self.get_stats()
            
            if not pairs: break

            best_pair = max(pairs, key=pairs.get)
            self.merge_pair(best_pair)

            print(f"Merge {i+1}: {best_pair}")
            
        self.final_vocab()
            
    def encode(self, text):
        tokens = []
        words = self.tokenize(text)

        for word in words:
            chars = list(word)
            i = 0
            while i < len(chars):
                matched = False        
                sorted_merges = sorted(self.merges, key=lambda x: len("".join(x)), reverse=True)
                for merge in sorted_merges:
                    merged = "".join(merge)
                    size = len(merged)
                    piece = "".join(chars[i:i+size])

                    if piece == merged:
                        tokens.append(merged)
                        i += size
                        matched = True
                        break

                if not matched:
                    tokens.append(chars[i])
                    i += 1

        encoded = [self.stoi["<sos>"]]
        for token in tokens:
            encoded.append(self.stoi.get(token, self.stoi["<unk>"]))

        encoded.append(self.stoi["<eos>"])

        return encoded
    
    def decode(self, token_ids):
        words = []

        for idx in token_ids:
            token = self.itos[idx]
            if token in ["<sos>", "<eos>", "<pad>"]: continue
            words.append(token)

        return " ".join(words)
    
# ================================================== SENTENCEPIECE TOKENIZER ==================================================
# =============================================================================================================================

"""
Encode: Raw text -> sentence pieces
Decode: Sentence pieces -> ids -> text
"""

import os
import tempfile
import sentencepiece as spm

from .tokenizer_base import TokenizerBase

class SentencePieceTokenizer(TokenizerBase):
    def __init__(self):
        super().__init__()
        self.sp = spm.SentencePieceProcessor()
        self.model_path = None

    def sos_id(self): return self.sp.bos_id()
    def eos_id(self): return self.sp.eos_id()

    def build_vocab(self, sentences, ctype="initial", ttype="BPE", lang="en", vocab_size=500):
        # Create temporary corpus file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as f:
            for sentence in sentences:
                f.write(sentence + "\n")
                
            corpus_path = f.name

        model_prefix = (f"vocabs/{ctype}_{ttype}_{lang}")

        try:
            spm.SentencePieceTrainer.train(
                input=corpus_path,
                model_prefix=model_prefix,
                vocab_size=vocab_size,
                model_type="bpe",
                normalization_rule_name="identity",
                pad_id=0,
                bos_id=1,
                eos_id=2,
                unk_id=3
            )

            self.model_path = (f"{model_prefix}.model")
            self.sp.load(self.model_path)
            self.vocab = [self.sp.id_to_piece(i) for i in range(self.sp.get_piece_size())]

        finally: os.remove(corpus_path)

    def encode(self, text):
        return self.sp.encode(text, out_type=int)

    def decode(self, ids):
        return self.sp.decode(ids)
    
    def id_to_piece(self, idx):
        return self.sp.id_to_piece(idx)
    
    def piece_to_id(self, piece):
        return self.sp.piece_to_id(piece)

    def save_vocab(self, path): pass


    def load_vocab(self, path):
        model_path = path.replace(".json", ".model")
        self.sp.load(model_path)
        self.vocab = [self.sp.id_to_piece(i) for i in range(self.sp.get_piece_size())]