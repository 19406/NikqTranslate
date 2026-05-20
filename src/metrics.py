import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def evaluate_model(model, loader, criterion, vi_tokenizer):
    model.eval()

    total_loss = 0
    bleu_scores = []

    smoothie = SmoothingFunction().method4

    with torch.no_grad():
        for src, tgt in loader:
            output = model(src, tgt)
            vocab_size = output.shape[-1]

            output_reshaped = output.reshape(-1, vocab_size)
            target = tgt[:, 1:].reshape(-1)

            loss = criterion(output_reshaped, target)
            total_loss += loss.item()

            predictions = model.translate(src, vi_tokenizer)

            # If batch size = 1
            if isinstance(predictions, str): predictions = [predictions]

            for pred, target_ids in zip(predictions, tgt):
                reference = vi_tokenizer.decode(target_ids.tolist()).split()
                candidate = pred.split()

                score = sentence_bleu(
                    [reference],
                    candidate,
                    smoothing_function=smoothie
                )

                bleu_scores.append(score)

    avg_loss = total_loss / len(loader)
    avg_bleu = sum(bleu_scores) / len(bleu_scores)

    return avg_loss, avg_bleu