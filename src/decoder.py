import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, attention):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention = attention
        self.lstm = nn.LSTM(embedding_dim + hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden, cell, encoder_outputs):
        embedded = self.embedding(x)
   
        attention_weights = self.attention(hidden, encoder_outputs).unsqueeze(1)
        context = torch.bmm(attention_weights, encoder_outputs)
        lstm_input = torch.cat((embedded, context), dim=2)
        
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        prediction = self.fc(output)

        return prediction, hidden, cell