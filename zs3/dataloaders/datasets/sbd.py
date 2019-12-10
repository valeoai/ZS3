import os
import pickle
import pathlib

import numpy as np
import scipy.io
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms

from zs3.dataloaders import custom_transforms as tr


SBD_DIR = pathlib.Path('./data/VOC2012/benchmark_RELEASE')


class SBDSegmentation(data.Dataset):
    NUM_CLASSES = 21

    def __init__(
        self,
        args,
        base_dir=SBD_DIR,
        split="train",
        load_embedding=None,
        w2c_size=300,
        weak_label=False,
        unseen_classes_idx_weak=[],
        transform=True,
    ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir
        self._dataset_dir = os.path.join(self._base_dir, "dataset")
        self._image_dir = os.path.join(self._dataset_dir, "img")
        self._cat_dir = os.path.join(self._dataset_dir, "cls")

        self.load_embedding = load_embedding
        self.w2c_size = w2c_size
        if self.load_embedding:
            self.init_embeddings()

        self.transform = transform
        self.weak_label = weak_label
        self.unseen_classes_idx_weak = unseen_classes_idx_weak

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.args = args
        # Get list of all images from the split and check that the files exist
        self.im_ids = []
        self.images = []
        self.categories = []
        for splt in self.split:
            with open(os.path.join(self._dataset_dir, splt + ".txt"), "r") as f:
                lines = f.read().splitlines()

            for line in lines:
                _image = os.path.join(self._image_dir, line + ".jpg")
                _categ = os.path.join(self._cat_dir, line + ".mat")
                assert os.path.isfile(_image)
                assert os.path.isfile(_categ)

                # if unseen classes
                if len(args.unseen_classes_idx) > 0:
                    _target = Image.fromarray(
                        scipy.io.loadmat(_categ)["GTcls"][0]["Segmentation"][0]
                    )
                    _target = np.array(_target, dtype=np.uint8)
                    if self.lbl_contains_unseen(_target, args.unseen_classes_idx):
                        continue

                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_categ)

        assert len(self.images) == len(self.categories)

        # Display stats
        print(f"(sbd) Number of images: {len(self.images):d}")

    def lbl_contains_unseen(self, lbl, unseen):
        unseen_pixel_mask = np.in1d(lbl.ravel(), unseen)
        if np.sum(unseen_pixel_mask) > 0:  # ignore images with any train_unseen pixels
            return True
        return False

    def init_embeddings(self):
        if self.load_embedding == "attributes":
            embed_arr = np.load("embeddings/pascal/pascalvoc_class_attributes.npy")
        elif self.load_embedding == "w2c":
            embed_arr = load_obj(
                "embeddings/pascal/w2c/norm_embed_arr_" + str(self.w2c_size)
            )
        elif self.load_embedding == "w2c_bg":
            embed_arr = np.load("embeddings/pascal/pascalvoc_class_w2c_bg.npy")
        elif self.load_embedding == "my_w2c":
            embed_arr = np.load("embeddings/pascal/pascalvoc_class_w2c.npy")
        elif self.load_embedding == "fusion":
            attributes = np.load("embeddings/pascal/pascalvoc_class_attributes.npy")
            w2c = np.load("embeddings/pascal/pascalvoc_class_w2c.npy")
            embed_arr = np.concatenate((attributes, w2c), axis=1)
        self.embeddings = torch.nn.Embedding(embed_arr.shape[0], embed_arr.shape[1])
        self.embeddings.weight.requires_grad = False
        self.embeddings.weight.data.copy_(torch.from_numpy(embed_arr))

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)

        if self.weak_label:
            unique_class = np.unique(np.array(_target))
            has_unseen_class = False
            for u_class in unique_class:
                if u_class in self.unseen_classes_idx_weak:
                    has_unseen_class = True
            if has_unseen_class:
                _target = Image.open(
                    "weak_label_pascal_10_unseen_top_by_image_25.0/sbd/"
                    + (self.categories[index].split("/"))[-1].split(".")[0]
                    + ".jpg"
                )

        sample = {"image": _img, "label": _target}

        if self.transform:
            sample = self.transform_s(sample)
        else:
            sample = self.transform_weak(sample)

        if self.load_embedding:
            mask = sample["label"] == 255
            sample["label"][mask] = 0
            lbl_vec = self.embeddings(sample["label"].long()).data
            lbl_vec = lbl_vec.permute(2, 0, 1)
            sample["label"][mask] = 255
            sample["label_emb"] = lbl_vec
        sample["image_name"] = self.images[index]
        return sample

    def __len__(self):
        return len(self.images)

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert("RGB")
        _target = Image.fromarray(
            scipy.io.loadmat(self.categories[index])["GTcls"][0]["Segmentation"][0]
        )

        return _img, _target

    def transform_s(self, sample):
        composed_transforms = transforms.Compose(
            [
                tr.RandomHorizontalFlip(),
                tr.RandomScaleCrop(
                    base_size=self.args.base_size, crop_size=self.args.crop_size
                ),
                tr.RandomGaussianBlur(),
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor(),
            ]
        )

        return composed_transforms(sample)

    def transform_weak(self, sample):

        composed_transforms = transforms.Compose(
            [
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor(),
            ]
        )

        return composed_transforms(sample)

    def __str__(self):
        return f"SBDSegmentation(split={self.split})"


def load_obj(name):
    with open(name + ".pkl", "rb") as f:
        return pickle.load(f, encoding="latin-1")
