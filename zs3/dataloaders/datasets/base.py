from torch.utils import data
import pickle
import torch
import numpy as np
from pathlib import Path


class BaseDataset(data.Dataset):
    def __init__(
        self,
        args,
        base_dir,
        split,
        load_embedding,
        w2c_size,
        weak_label,
        unseen_classes_idx_weak,
        transform,
    ):
        super().__init__()
        self.args = args
        self._base_dir = Path(base_dir)
        self.split = split
        self.load_embedding = load_embedding
        self.w2c_size = w2c_size
        self.embeddings = None
        if self.load_embedding:
            self.init_embeddings()
        self.images = []
        self.weak_label = weak_label
        self.unseen_classes_idx_weak = unseen_classes_idx_weak
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def init_embeddings(self):
        raise NotImplementedError

    def make_embeddings(self, embed_arr):
        self.embeddings = torch.nn.Embedding(embed_arr.shape[0], embed_arr.shape[1])
        self.embeddings.weight.requires_grad = False
        self.embeddings.weight.data.copy_(torch.from_numpy(embed_arr))

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


def lbl_contains_unseen(lbl, unseen):
    unseen_pixel_mask = np.in1d(lbl.ravel(), unseen)
    if np.sum(unseen_pixel_mask) > 0:  # ignore images with any train_unseen pixels
        return True
    return False
