import torch
from src import i2T, build_seq2seq

EMBEDDING_DIM = 32
HIDDEN_DIM = 64

model, en_tokenizer, vi_tokenizer = build_seq2seq("data/corpus.json", EMBEDDING_DIM, HIDDEN_DIM)
model.load_state_dict(torch.load("models/seq2seq.pt"))

def translation():
    while True:
        sentence = input("Enter a sentence: ")
        if sentence == "exit": return
        src = i2T([en_tokenizer.encode(sentence)])
        translation = model.translate(src, vi_tokenizer)

        print("Translation:", translation)