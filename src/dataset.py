from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, src_sequences, tgt_sequences):
        self.src = src_sequences
        self.tgt = tgt_sequences

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return (self.src[idx], self.tgt[idx])