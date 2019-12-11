import pathlib

import numpy as np
from PIL import Image
from torchvision import transforms

from zs3.dataloaders import custom_transforms as tr
from .base import BaseDataset, load_obj, lbl_contains_unseen


PASCAL_DIR = pathlib.Path("./data/VOC2012")


class VOCSegmentation(BaseDataset):
    """
    PascalVoc dataset
    """
    NUM_CLASSES = 21

    def __init__(
        self,
        args,
        base_dir=PASCAL_DIR,
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
        super().__init__(
            args,
            base_dir,
            split,
            load_embedding,
            w2c_size,
            weak_label,
            unseen_classes_idx_weak,
            transform,
        )
        self._image_dir = self._base_dir / "JPEGImages"
        self._cat_dir = self._base_dir / "SegmentationClass"

        self.unseen_classes_idx_weak = unseen_classes_idx_weak

        _splits_dir = self._base_dir / "ImageSets" / "Segmentation"

        self.im_ids = []
        self.categories = []

        lines = (_splits_dir / f"{self.split}.txt").read_text().splitlines()

        for ii, line in enumerate(lines):
            _image = self._image_dir / f"{line}.jpg"
            _cat = self._cat_dir / f"{line}.png"
            assert _image.is_file(), _image
            assert _cat.is_file(), _cat

            # if unseen classes and training split
            if len(args.unseen_classes_idx) > 0 and self.split == "train":
                cat = Image.open(_cat)
                cat = np.array(cat, dtype=np.uint8)
                if lbl_contains_unseen(cat, args.unseen_classes_idx):
                    continue

            self.im_ids.append(line)
            self.images.append(_image)
            self.categories.append(_cat)

        assert len(self.images) == len(self.categories)

        # Display stats
        print(f"(pascal) Number of images in {split}: {len(self.images):d}")

    def init_embeddings(self):
        embed_arr = load_obj("embeddings/pascal/w2c/norm_embed_arr_" + str(self.w2c_size))
        self.make_embeddings(embed_arr)

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
                    "weak_label_pascal_10_unseen_top_by_image_25.0/pascal/"
                    + self.categories[index].stem
                    + ".jpg"
                )

        sample = {"image": _img, "label": _target}

        if self.transform:
            if self.split == "train":
                sample = self.transform_tr(sample)
            elif self.split == "val":
                sample = self.transform_val(sample)
        else:
            sample = self.transform_weak(sample)

        if self.load_embedding:
            self.get_embeddings(sample)
        sample["image_name"] = str(self.images[index])
        return sample

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert("RGB")
        _target = Image.open(self.categories[index])
        return _img, _target

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose(
            [
                tr.RandomHorizontalFlip(),
                tr.RandomScaleCrop(
                    base_size=self.args.base_size,
                    crop_size=self.args.crop_size,
                    fill=255,
                ),
                tr.RandomGaussianBlur(),
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor(),
            ]
        )
        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose(
            [
                tr.FixScale(crop_size=self.args.crop_size),
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
        return f"VOC2012(split={self.split})"
