from torch.utils import data
import pickle


class BaseDataset(data.Dataset):

    def __init__(self):
        super().__init__()
        self.images = []

    def __len__(self):
        return len(self.images)


def load_obj(name):
    with open(name + ".pkl", "rb") as f:
        return pickle.load(f, encoding="latin-1")
