from torch.utils.data import Dataset


class PlainTextDataset(Dataset):
    def __init__(self, text_path):
        with open(text_path, 'r') as f:
            self.texts = [sentence.rstrip() for sentence in f.readlines()]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        return self.texts[item]