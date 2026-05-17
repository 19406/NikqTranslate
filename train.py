import torch
import torch.nn as nn
import torch.optim as optim

from src import create_token_ids, build_seq2seq

torch.manual_seed(42)

EMBEDDING_DIM = 32
HIDDEN_DIM = 64
EPOCHS = 10000

print("Building Seq2Seq model...")
model, en_tokenizer, vi_tokenizer = build_seq2seq("data/corpus.json", EMBEDDING_DIM, HIDDEN_DIM)
print("Model initialized!")

print(f"English vocab size: {len(en_tokenizer.vocab)}")
print(f"Vietnamese vocab size: {len(vi_tokenizer.vocab)}")

print("\nEncoding source sentences...")
src = create_token_ids("data/corpus.json", lang="en")

print("Encoding target sentences...")
tgt = create_token_ids("data/corpus.json", lang="vi")

print("\nTensor shapes:")
print(f"src shape: {src.shape}")
print(f"tgt shape: {tgt.shape}")

# Loss function
criterion = nn.CrossEntropyLoss(ignore_index=0)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_loop():
    print("\nTraining started...\n")
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        output = model(src, tgt)
        vocab_size = output.shape[-1]

        if epoch == 0:
            print("Forward pass successful!")
            print(f"Raw output shape: {output.shape}")
            print(f"Vocabulary size: {vocab_size}")

        output = output.reshape(-1, vocab_size)
        target = tgt[:, 1:].reshape(-1)

        if epoch == 0:
            print(f"Reshaped output shape: {output.shape}")
            print(f"Target shape: {target.shape}")

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch + 1}/{EPOCHS}] | Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "models/seq2seq.pt")

    print("\nTraining complete!")