import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from src import TranslationDataset, collate_func, create_token_ids, build_seq2seq, evaluate_model

import json
import matplotlib.pyplot as plt

torch.manual_seed(42)

TOKENIZER_TYPE = "BPE"
EMBEDDING_DIM = 32
HIDDEN_DIM = 64
LEARNING_RATE = 0.001
EPOCHS = 1000

def initialize_model(rebuild_vocab=False):
    print("Building Seq2Seq model...")
    model, en_tokenizer, vi_tokenizer = build_seq2seq("data/corpus.json", TOKENIZER_TYPE, rebuild_vocab, EMBEDDING_DIM, HIDDEN_DIM)
    if rebuild_vocab:
        en_tokenizer.save_vocab(f"vocabs/{TOKENIZER_TYPE}_en_vocab.json")
        vi_tokenizer.save_vocab(f"vocabs/{TOKENIZER_TYPE}_vi_vocab.json")

    print("Model initialized!")

    print(f"English vocab size: {len(en_tokenizer.vocab)}")
    print(f"Vietnamese vocab size: {len(vi_tokenizer.vocab)}")

    print("\nEncoding source sentences...")
    src_sequences = create_token_ids("data/corpus.json", TOKENIZER_TYPE, lang="en")

    print("Encoding target sentences...")
    tgt_sequences = create_token_ids("data/corpus.json", TOKENIZER_TYPE, lang="vi")

    dataset = TranslationDataset(src_sequences, tgt_sequences)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=True,
            collate_fn=collate_func
        )
    
    val_loader = DataLoader(
            val_dataset,
            batch_size=8,
            shuffle=True,
            collate_fn=collate_func
        )

    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    return model, train_loader, val_loader, criterion, optimizer, vi_tokenizer
    

def train_loop(rebuild_vocab=False):
    model, train_loader, val_loader, criterion, optimizer, vi_tokenizer = initialize_model(rebuild_vocab)
    
    print("\nTraining started...\n")
    
    train_loss_history = []
    val_loss_history = []
    val_bleu_history = []
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for batch_idx, (src, tgt) in enumerate(train_loader):            
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
            print(f"Epoch [{epoch + 1}/{EPOCHS}] | Batch [{batch_idx + 1}/{len(train_loader)}] | Loss: {loss.item():.4f}")

        avg_train_loss = epoch_loss / len(train_loader)
        val_loss, val_bleu = evaluate_model(model, val_loader, criterion, vi_tokenizer)
        
        train_loss_history.append(avg_train_loss)
        val_loss_history.append(val_loss)
        val_bleu_history.append(val_bleu)
        
        print(f"Epoch [{epoch + 1}/{EPOCHS}]")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val BLEU: {val_bleu:.4f}")
    
    metrics = {
        "train_loss": train_loss_history,
        "val_loss": val_loss_history,
        "val_bleu": val_bleu_history
    }    
    with open(f"logs/{TOKENIZER_TYPE}_metrics_history.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 10))
    axes[0].plot(train_loss_history, label="Train Loss")
    axes[0].plot(val_loss_history, label="Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(val_bleu_history, label="Validation BLEU")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("BLEU")
    axes[1].set_title("BLEU Curves")
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"plots/{TOKENIZER_TYPE}_metrics_curve.png")
    plt.show()

    torch.save(model.state_dict(), "models/seq2seq.pt")

    print("\nTraining complete!")