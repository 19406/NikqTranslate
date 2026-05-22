import torch
import torch.nn as nn

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt):
        encoder_outputs, hidden, cell = self.encoder(src)
        outputs = []
        input_token = tgt[:, 0].unsqueeze(1)

        for t in range(1, tgt.shape[1]):
            prediction, hidden, cell = self.decoder(input_token, hidden, cell, encoder_outputs)
            outputs.append(prediction)
            input_token = tgt[:, t].unsqueeze(1)

        outputs = torch.cat(outputs, dim=1)

        return outputs
    
    def translate(self, src, vi_tokenizer, max_length=50):
        self.eval()
        
        with torch.no_grad():
            if src.dim() == 1: src = src.unsqueeze(0)
            
            batch_size = src.size(0)
            
            encoder_outputs, hidden, cell = self.encoder(src)
            input_token = torch.full(
                (batch_size, 1),
                vi_tokenizer.sos_id(),
                dtype=torch.long,
                device=src.device
            )

            generated_tokens = [[] for _ in range(batch_size)]
            finished = [False for _ in range(batch_size)]
            
            for _ in range(max_length):
                prediction, hidden, cell = self.decoder(input_token, hidden, cell, encoder_outputs)
                predicted_token = torch.argmax(prediction, dim=-1)

                input_token = predicted_token
                
                for i in range(batch_size):
                    if finished[i]: continue
                    
                    token_id = predicted_token[i].item()
                    if token_id == vi_tokenizer.eos_id():
                        finished[i] = True
                        continue

                    generated_tokens[i].append(token_id)
                
                if all(finished): break
                
            decoded_sentences = []
            
            for tokens in generated_tokens:
                sentence = vi_tokenizer.decode(tokens)
                decoded_sentences.append(sentence)
                
            if batch_size == 1: return decoded_sentences[0]
            return decoded_sentences