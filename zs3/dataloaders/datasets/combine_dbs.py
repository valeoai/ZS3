import torch.utils.data as data


class CombineDBs(data.Dataset):
    NUM_CLASSES = 21

    def __init__(self, dataloaders, excluded=None):
        self.dataloaders = dataloaders
        self.excluded = excluded
        self.im_ids = []

        # Combine object lists
        for dl in dataloaders:
            for elem in dl.im_ids:
                if elem not in self.im_ids:
                    self.im_ids.append(elem)

        # Exclude
        if excluded:
            for dl in excluded:
                for elem in dl.im_ids:
                    if elem in self.im_ids:
                        self.im_ids.remove(elem)

        # Get object pointers
        self.cat_list = []
        new_im_ids = []
        num_images = 0
        for ii, dl in enumerate(dataloaders):
            for jj, curr_im_id in enumerate(dl.im_ids):
                if (curr_im_id in self.im_ids) and (curr_im_id not in new_im_ids):
                    num_images += 1
                    new_im_ids.append(curr_im_id)
                    self.cat_list.append({"db_ii": ii, "cat_ii": jj})

        self.im_ids = new_im_ids
        print(f"Combined number of images: {num_images:d}")

    def __getitem__(self, index):

        _db_ii = self.cat_list[index]["db_ii"]
        _cat_ii = self.cat_list[index]["cat_ii"]
        sample = self.dataloaders[_db_ii].__getitem__(_cat_ii)

        if "meta" in sample.keys():
            sample["meta"]["db"] = str(self.dataloaders[_db_ii])

        return sample

    def __len__(self):
        return len(self.cat_list)

    def __str__(self):
        include_db = [str(db) for db in self.dataloaders]
        exclude_db = [str(db) for db in self.excluded]
        return (
            "Included datasets:"
            + str(include_db)
            + "\n"
            + "Excluded datasets:"
            + str(exclude_db)
        )
