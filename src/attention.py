import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.energy = nn.Linear(hidden_dim * 2, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[0]
        seq_len = encoder_outputs.shape[1]

        hidden = hidden.permute(1, 0, 2)
        hidden = hidden.repeat(1, seq_len, 1)

        energy = torch.tanh(self.energy(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.score(energy).squeeze(2)

        return torch.softmax(attention, dim=1)