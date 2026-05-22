import torch
from src import i2T, build_seq2seq

CORPUS_TYPE = "align"
TOKENIZER_TYPE = "BPE"
EMBEDDING_DIM = 32
HIDDEN_DIM = 64

model, en_tokenizer, vi_tokenizer = build_seq2seq("data/corpus.json", CORPUS_TYPE, TOKENIZER_TYPE, False, EMBEDDING_DIM, HIDDEN_DIM)
model.load_state_dict(torch.load(f"models/{CORPUS_TYPE}_{TOKENIZER_TYPE}_seq2seq.pt"))

def translation():
    while True:
        sentence = input("Enter a sentence: ")
        if sentence == "exit": return
        src = i2T([en_tokenizer.encode(sentence)])
        translation = model.translate(src, vi_tokenizer)

        print("Translation:", translation)