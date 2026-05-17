import torch
from src import Encoder, Decoder, Attention, build_vocab, i2T, T2i

EMBEDDING_DIM = 32
HIDDEN_DIM = 64

def test_encoder():
    tokenizer = build_vocab("data/corpus.json", lang="en")
    vocab_size = len(tokenizer.vocab)
    encoder = Encoder(vocab_size, EMBEDDING_DIM, HIDDEN_DIM)    
    
    text = input("Enter text: ")
    encoded = tokenizer.encode(text)
    outputs, hidden, cell = encoder(i2T(encoded))

    print("Outputs shape:", outputs.shape)
    print("Hidden state shape:", hidden.shape)
    print("Cell state shape:", cell.shape)
    print("Outputs:", outputs)
    print("Hidden state:", hidden)
    print("Cell state:", cell)
    
def test_decoder():
    tokenizer = build_vocab("data/corpus.json", lang="en")
    vocab_size = len(tokenizer.vocab)
    encoder = Encoder(vocab_size, EMBEDDING_DIM, HIDDEN_DIM)
    
    attention = Attention(HIDDEN_DIM)
    decoder = Decoder(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, attention)    
    
    text = input("Enter text: ")
    encoded = [tokenizer.encode(text)]
    encoder_outputs, hidden, cell = encoder(i2T(encoded))
    
    input_token = i2T([[encoded[0][0]]])

    prediction, hidden, cell = decoder(input_token, hidden, cell, encoder_outputs)
    
    print("Prediction shape:", prediction.shape)
    print("Prediction:", prediction)
    print("Max prediction:", torch.max(prediction, dim=-1))
    print("Output sentence:", tokenizer.decode(T2i(torch.argmax(prediction, dim=-1).squeeze(1))))