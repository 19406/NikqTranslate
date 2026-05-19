import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src import TranslationDataset, collate_func, create_token_ids, build_seq2seq

torch.manual_seed(42)

TOKENIZER_TYPE = "BPE"
EMBEDDING_DIM = 32
HIDDEN_DIM = 64
LEARNING_RATE = 0.001
EPOCHS = 100

def initialize_model(rebuild_vocab=False):
    print("Building Seq2Seq model...")
    model, en_tokenizer, vi_tokenizer = build_seq2seq("data/corpus.json", TOKENIZER_TYPE, rebuild_vocab, EMBEDDING_DIM, HIDDEN_DIM)
    if rebuild_vocab:
        en_tokenizer.save_vocab("vocabs/en_vocab.json")
        vi_tokenizer.save_vocab("vocabs/vi_vocab.json")

    print("Model initialized!")

    print(f"English vocab size: {len(en_tokenizer.vocab)}")
    print(f"Vietnamese vocab size: {len(vi_tokenizer.vocab)}")

    print("\nEncoding source sentences...")
    src_sequences = create_token_ids("data/corpus.json", TOKENIZER_TYPE, lang="en")

    print("Encoding target sentences...")
    tgt_sequences = create_token_ids("data/corpus.json", TOKENIZER_TYPE, lang="vi")

    dataset = TranslationDataset(src_sequences, tgt_sequences)
    loader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=True,
            collate_fn=collate_func
        )

    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    return model, loader, criterion, optimizer
    

def train_loop(rebuild_vocab=False):
    model, loader, criterion, optimizer = initialize_model(rebuild_vocab)
    
    print("\nTraining started...\n")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for batch_idx, (src, tgt) in enumerate(loader):            
            optimizer.zero_grad()
            output = model(src, tgt)
            vocab_size = output.shape[-1]

            output = output.reshape(-1, vocab_size)
            target = tgt[:, 1:].reshape(-1)

            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            print(f"Epoch [{epoch + 1}/{EPOCHS}] | Batch [{batch_idx + 1}/{len(loader)}] | Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch [{epoch + 1}/{EPOCHS}] - Average Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "models/seq2seq.pt")

    print("\nTraining complete!")