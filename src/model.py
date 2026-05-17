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
    
    def translate(self, src, vi_tokenizer, max_length=20):
        self.eval()
        
        with torch.no_grad():
            encoder_outputs, hidden, cell = self.encoder(src)
            input_token = torch.tensor([[1]])

            generated_tokens = []
            for _ in range(max_length):
                prediction, hidden, cell = self.decoder(input_token, hidden, cell, encoder_outputs)
                predicted_token = prediction.argmax(dim=-1)

                token_id = predicted_token.item()
                if token_id == vi_tokenizer.stoi["<eos>"]: break

                generated_tokens.append(token_id)
                input_token = predicted_token

            return vi_tokenizer.decode(generated_tokens)