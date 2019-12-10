from torch.utils import data
import pickle


class BaseDataset(data.Dataset):

    def __init__(self):
        super().__init__()
        self.images = []

    def __len__(self):
        return len(self.images)

    def get_embeddings(self, sample):
        mask = sample["label"] == 255
        sample["label"][mask] = 0
        lbl_vec = self.embeddings(sample["label"].long()).data
        lbl_vec = lbl_vec.permute(2, 0, 1)
        sample["label"][mask] = 255
        sample["label_emb"] = lbl_vec


def load_obj(name):
    with open(name + ".pkl", "rb") as f:
        return pickle.load(f, encoding="latin-1")
