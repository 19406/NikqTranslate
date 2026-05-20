from .tokenizer import SimpleTokenizer, BPETokenizer
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

def collate_func(batch):
    src_batch = []
    tgt_batch = []
    for src, tgt in batch:
        src_batch.append(src)
        tgt_batch.append(tgt)

    src_max_len = max(len(seq) for seq in src_batch)
    tgt_max_len = max(len(seq) for seq in tgt_batch)

    padded_src = []
    padded_tgt = []
    for seq in src_batch:
        padded = seq + [0] * (src_max_len - len(seq))
        padded_src.append(padded)

    for seq in tgt_batch:
        padded = seq + [0] * (tgt_max_len - len(seq))
        padded_tgt.append(padded)

    return (i2T(padded_src), i2T(padded_tgt))

def create_token_ids(data_path, ttype="BPE", lang="en"):
    sentences = parse_sentences(data_path, lang)
    tokenizer = build_vocab(data_path, ttype, False, lang)
    
    encoded = [tokenizer.encode(s) for s in sentences]
    return encoded

def build_vocab(data_path, ttype="BPE", rebuild_vocab=False, lang="en"):
    sentences = parse_sentences(data_path, lang)
    
    if ttype == "simple": tokenizer = SimpleTokenizer()
    elif ttype == "BPE": tokenizer = BPETokenizer()
    
    if rebuild_vocab: tokenizer.build_vocab(sentences)
    else: tokenizer.load_vocab(f"vocabs/{ttype}_{lang}_vocab.json")
    
    return tokenizer

def i2T(int_list):
    return torch.tensor(int_list, dtype=torch.long)

def T2i(tensor):
    return tensor.tolist()

def build_seq2seq(data_path, ttype="BPE", rebuild_vocab=False, embedding_dim=32, hidden_dim=64):
    en_tokenizer = build_vocab(data_path, ttype, rebuild_vocab, lang="en")
    vi_tokenizer = build_vocab(data_path, ttype, rebuild_vocab, lang="vi")
    en_vocab_size = len(en_tokenizer.vocab)
    vi_vocab_size = len(vi_tokenizer.vocab)

    encoder = Encoder(en_vocab_size, embedding_dim, hidden_dim)
    attention = Attention(hidden_dim)
    decoder = Decoder(vi_vocab_size, embedding_dim, hidden_dim, attention)
    model = Seq2Seq(encoder, decoder)
    
    return model, en_tokenizer, vi_tokenizer