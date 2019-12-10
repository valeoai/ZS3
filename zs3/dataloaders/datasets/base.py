from torch.utils import data


class BaseDataset(data.Dataset):

    def __init__(self):
        super().__init__()
        self.images = []

    def __len__(self):
        return len(self.images)
