from .tokenizer import Tokenizer
from .encoder import Encoder
from .decoder import Decoder
from .attention import Attention
from .model import Seq2Seq

import json
import torch

def pad_sequences(sequences, pad_idx=0):
    max_len = max(len(seq) for seq in sequences)

    padded = []
    for seq in sequences:
        seq = seq + ([pad_idx] * (max_len - len(seq)))
        padded.append(seq)

    return padded

def parse_sentences(data_path, lang="en"):
    with open(data_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    sentences = []
    for pair in pairs: sentences.append(pair[lang])
    return sentences

def create_token_ids(data_path, lang="en"):
    sentences = parse_sentences(data_path, lang)
    tokenizer = build_vocab(data_path, lang)
    
    encoded = [tokenizer.encode(s) for s in sentences]
    padded = pad_sequences(encoded, pad_idx=tokenizer.stoi["<pad>"])
    return i2T(padded) 

def build_vocab(data_path, lang="en"):
    sentences = parse_sentences(data_path, lang)

    tokenizer = Tokenizer()
    tokenizer.build_vocab(sentences)
    
    return tokenizer

def i2T(int_list):
    return torch.tensor(int_list, dtype=torch.long)

def T2i(tensor):
    return tensor.tolist()

def build_seq2seq(data_path, embedding_dim=32, hidden_dim=64):
    en_tokenizer = build_vocab(data_path, lang="en")
    vi_tokenizer = build_vocab(data_path, lang="vi")
    en_vocab_size = len(en_tokenizer.vocab)
    vi_vocab_size = len(vi_tokenizer.vocab)

    encoder = Encoder(en_vocab_size, embedding_dim, hidden_dim)
    attention = Attention(hidden_dim)
    decoder = Decoder(vi_vocab_size, embedding_dim, hidden_dim, attention)
    model = Seq2Seq(encoder, decoder)
    
    return model, en_tokenizer, vi_tokenizer