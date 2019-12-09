import os
import os.path as osp
import pickle

import numpy as np
import scipy
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from zs3.dataloaders import custom_transforms as tr
from zs3.mypath import Path


class ContextSegmentation(Dataset):
    class_names = [
        "background",  # class 0
        "aeroplane",  # class 1
        "bicycle",  # class 2
        "bird",  # class 3
        "boat",  # class 4
        "bottle",  # class 5
        "bus",  # class 6
        "car",  # class 7
        "cat",  # class 8
        "chair",  # class 9
        "cow",  # class 10
        "table",  # class 11
        "dog",  # class 12
        "horse",  # class 13
        "motorbike",  # class 14
        "person",  # class 15
        "pottedplant",  # class 16
        "sheep",  # class 17
        "sofa",  # class 18
        "train",  # class 19
        "tvmonitor",  # class 20
        "bag",  # class 21
        "bed",  # class 22
        "bench",  # class 23
        "book",  # class 24
        "building",  # class 25
        "cabinet",  # class 26
        "ceiling",  # class 27
        "cloth",  # class 28
        "computer",  # class 29
        "cup",  # class 30
        "door",  # class 31
        "fence",  # class 32
        "floor",  # class 33
        "flower",  # class 34
        "food",  # class 35
        "grass",  # class 36
        "ground",  # class 37
        "keyboard",  # class 38
        "light",  # class 39
        "mountain",  # class 40
        "mouse",  # class 41
        "curtain",  # class 42
        "platform",  # class 43
        "sign",  # class 44
        "plate",  # class 45
        "road",  # class 46
        "rock",  # class 47
        "shelves",  # class 48
        "sidewalk",  # class 49
        "sky",  # class 50
        "snow",  # class 51
        "bedclothes",  # class 52
        "track",  # class 53
        "tree",  # class 54
        "truck",  # class 55
        "wall",  # class 56
        "water",  # class 57
        "window",  # class 58
        "wood",  # class 59
    ]

    """
    PascalVoc dataset
    """
    NUM_CLASSES = 60

    def __init__(
        self,
        args,
        base_dir=Path.db_root_dir("context"),
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

        self.transform = transform

        self._image_dir = os.path.join(
            self._base_dir, "pascal/VOCdevkit/VOC2012/JPEGImages"
        )
        self._cat_dir = os.path.join(self._base_dir, "full_annotations/trainval")

        self.weak_label = weak_label
        self.unseen_classes_idx_weak = unseen_classes_idx_weak

        self.split = split

        self.args = args

        self.load_embedding = load_embedding
        self.w2c_size = w2c_size
        if self.load_embedding:
            self.init_embeddings()

        _splits_dir = os.path.join(self._base_dir)

        self.im_ids = []
        self.images = []
        self.categories = []

        self.labels_459 = [
            label.decode().replace(" ", "")
            for idx, label in np.genfromtxt(
                osp.join(self._base_dir, "full_annotations/labels.txt"),
                delimiter=":",
                dtype=None,
            )
        ]
        self.labels_59 = [
            label.decode().replace(" ", "")
            for idx, label in np.genfromtxt(
                osp.join(self._base_dir, "classes-59.txt"), delimiter=":", dtype=None
            )
        ]
        for main_label, task_label in zip(
            ("table", "bedclothes", "cloth"), ("diningtable", "bedcloth", "clothes")
        ):
            self.labels_59[self.labels_59.index(task_label)] = main_label

        self.idx_59_to_idx_469 = {}
        for idx, l in enumerate(self.labels_59):
            if idx > 0:
                self.idx_59_to_idx_469[idx] = self.labels_459.index(l) + 1

        with open(os.path.join(os.path.join(_splits_dir, self.split + ".txt")), "r") as f:
            lines = f.read().splitlines()

        for ii, line in enumerate(lines):
            _image = os.path.join(self._image_dir, line + ".jpg")
            _cat = os.path.join(self._cat_dir, line + ".mat")
            assert os.path.isfile(_image)
            assert os.path.isfile(_cat)

            # if unseen classes and training split
            if len(args.unseen_classes_idx) > 0:
                cat = self.load_label(_cat)
                if self.lbl_contains_unseen(cat, args.unseen_classes_idx):
                    continue

            self.im_ids.append(line)
            self.images.append(_image)
            self.categories.append(_cat)

        assert len(self.images) == len(self.categories)

        # Display stats
        print(
            "(pascal) Number of images in {}: {:d}, {:d} deleted".format(
                split, len(self.images), len(lines) - len(self.images)
            )
        )

    def load_label(self, file_path):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        The full 459 labels are translated to the 59 class task labels.
        """
        label_459 = scipy.io.loadmat(file_path)["LabelMap"]
        label = np.zeros_like(label_459, dtype=np.uint8)
        for idx, l in enumerate(self.labels_59):
            if idx > 0:
                label[label_459 == self.idx_59_to_idx_469[idx]] = idx
        return label

    def lbl_contains_unseen(self, lbl, unseen):
        unseen_pixel_mask = np.in1d(lbl.ravel(), unseen)
        if np.sum(unseen_pixel_mask) > 0:  # ignore images with any train_unseen pixels
            return True
        return False

    def init_embeddings(self):
        if self.load_embedding == "my_w2c":
            embed_arr = np.load("embeddings/context/pascalcontext_class_w2c.npy")
        self.embeddings = torch.nn.Embedding(embed_arr.shape[0], embed_arr.shape[1])
        self.embeddings.weight.requires_grad = False
        self.embeddings.weight.data.copy_(torch.from_numpy(embed_arr))

    def __len__(self):
        return len(self.images)

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
                    "weak_label_context_10_unseen_top_by_image_75.0/pascal/"
                    + (self.categories[index].split("/"))[-1].split(".")[0]
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
            mask = sample["label"] == 255
            sample["label"][mask] = 0
            lbl_vec = self.embeddings(sample["label"].long()).data
            lbl_vec = lbl_vec.permute(2, 0, 1)
            sample["label"][mask] = 255
            sample["label_emb"] = lbl_vec
            # sample = {'image': sample['image'], 'label': sample['label'], 'label_emb': lbl_vec}
        sample["image_name"] = self.images[index]
        return sample

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert("RGB")
        _target = self.load_label(self.categories[index])
        _target = Image.fromarray(_target)
        return _img, _target

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose(
            [
                tr.RandomHorizontalFlip(),
                tr.RandomScaleCrop(
                    base_size=self.args.base_size, crop_size=self.args.crop_size, fill=255
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
        return "VOC2012(split=" + str(self.split) + ")"


if __name__ == "__main__":
    from zs3.dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument(
        "--out-stride", type=int, default=16, help="network output stride (default: 8)"
    )

    # PASCAL VOC
    parser.add_argument(
        "--dataset",
        type=str,
        default="context",
        choices=["pascal", "coco", "cityscapes"],
        help="dataset name (default: pascal)",
    )

    parser.add_argument(
        "--use-sbd",
        action="store_true",
        default=True,
        help="whether to use SBD dataset (default: True)",
    )
    parser.add_argument(
        "--workers", type=int, default=4, metavar="N", help="dataloader threads"
    )
    parser.add_argument("--base-size", type=int, default=312, help="base image size")
    parser.add_argument("--crop-size", type=int, default=312, help="crop image size")

    parser.add_argument(
        "--sync-bn",
        type=bool,
        default=None,
        help="whether to use sync bn (default: auto)",
    )
    parser.add_argument(
        "--freeze-bn",
        type=bool,
        default=False,
        help="whether to freeze bn parameters (default: False)",
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        default="ce",
        choices=["ce", "focal"],
        help="loss func type (default: ce)",
    )
    # training hyper params

    # PASCAL VOC
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        metavar="N",
        help="number of epochs to train (default: auto)",
    )

    parser.add_argument(
        "--start_epoch", type=int, default=0, metavar="N", help="start epochs (default:0)"
    )

    # PASCAL VOC
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        metavar="N",
        help="input batch size for training (default: auto)",
    )

    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        metavar="N",
        help="input batch size for \
                                    testing (default: auto)",
    )
    parser.add_argument(
        "--use-balanced-weights",
        action="store_true",
        default=False,
        help="whether to use balanced weights (default: False)",
    )

    # optimizer params
    # PASCAL VOC
    parser.add_argument(
        "--lr",
        type=float,
        default=0.007,
        metavar="LR",
        help="learning rate (default: auto)",
    )

    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="poly",
        choices=["poly", "step", "cos"],
        help="lr scheduler mode: (default: poly)",
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, metavar="M", help="momentum (default: 0.9)"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=5e-4,
        metavar="M",
        help="w-decay (default: 5e-4)",
    )
    parser.add_argument(
        "--nesterov",
        action="store_true",
        default=False,
        help="whether use nesterov (default: False)",
    )
    # cuda, seed and logging
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="0",
        help="use which gpu to train, must be a \
                            comma-separated list of integers only (default=0)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    # checking point
    parser.add_argument(
        "--resume",
        type=str,
        default="/home/docker_user/workspace/zero-shot_object_detection/segmentation_pascalvoc_deeplabv3/run/context/context_full/experiment_3/checkpoint.pth.tar",
        help="put the path to resuming file if needed",
    )

    # parser.add_argument('--resume', type=str, default='/home/docker_user/workspace/zero-shot_object_detection/segmentation_pascalvoc_deeplabv3/run/pascal/pascal_only_seen_10_wo_imagenet_is312/experiment_0/1000_model.pth.tar')

    # parser.add_argument('--resume', type=str, default='/home/docker_user/workspace/zero-shot_object_detection/segmentation_pascalvoc_deeplabv3/run/pascal_final/deeplab-resnet/pascal_only_seen_embedding300_myw2c_cosine/experiment_2/270_model.pth.tar')

    # parser.add_argument('--resume', type=str, default='/home/docker_user/workspace/zero-shot_object_detection/segmentation_pascalvoc_deeplabv3/run/pascal_final/deeplab-resnet/pascal_only_seen_embedding20_myw2c_cosine/experiment_0/475_model.pth.tar')
    # parser.add_argument('--resume', type=str, default='/home/docker_user/workspace/zero-shot_object_detection/segmentation_pascalvoc_deeplabv3/run/pascal_final/deeplab-resnet/pascal_only_seen_wo_imagenet/experiment_0/1000_model.pth.tar', help='put the path to resuming file if needed')

    # parser.add_argument('--resume', type=str, default='/home/docker_user/workspace/zero-shot_object_detection/segmentation_pascalvoc_deeplabv3/run/coco_final/nopascalclasses/experiment_0/checkpoint.pth.tar')

    # parser.add_argument('--resume', type=str,
    #                    default='/home/docker_user/workspace/zero-shot_object_detection/segmentation_pascalvoc_deeplabv3/run/coco/nopascalclasses/256_model.pth.tar',
    #                    help='put the path to resuming file if needed')

    parser.add_argument(
        "--checkname",
        type=str,
        default="gmmn_pascal_w2c300_linear_weighted100_hs256_context_aware",
    )

    # false if embedding resume
    parser.add_argument("--global_avg_pool_bn", type=bool, default=True)

    # finetuning pre-trained models
    parser.add_argument(
        "--ft",
        action="store_true",
        default=False,
        help="finetuning on a different dataset",
    )
    # evaluation option
    parser.add_argument(
        "--eval-interval", type=int, default=1, help="evaluation interval (default: 1)"
    )
    parser.add_argument(
        "--no-val",
        action="store_true",
        default=False,
        help="skip validation during training",
    )

    ### FOR IMAGE SELECTION IN ORDER TO TAKE OFF IMAGE WITH UNSEEN CLASSES FOR TRAINING
    # all classes
    parser.add_argument("--unseen_classes_idx", type=int, default=[10, 14])
    # only seen classes
    # parser.add_argument('--unseen_classes_idx', type=int, default=[10, 14, 16])

    ### FOR METRIC COMPUTATION IN ORDER TO GET PERFORMANCES FOR TWO SETS
    seen_classes_idx_metric = np.arange(21)
    unseen_classes_idx_metric = [10, 14]

    # unseen_classes_idx_metric = [10, 14, 12, 1, 9, 3, 13, 7, 5, 20]

    seen_classes_idx_metric = np.delete(
        seen_classes_idx_metric, unseen_classes_idx_metric
    ).tolist()
    parser.add_argument(
        "--seen_classes_idx_metric", type=int, default=seen_classes_idx_metric
    )
    parser.add_argument(
        "--unseen_classes_idx_metric", type=int, default=unseen_classes_idx_metric
    )

    parser.add_argument(
        "--unseen_weight", type=int, default=100, help="number of output channels"
    )

    parser.add_argument(
        "--nonlinear_last_layer", type=bool, default=False, help="non linear prediction"
    )
    parser.add_argument(
        "--random_last_layer", type=bool, default=True, help="randomly init last layer"
    )

    parser.add_argument(
        "--real_seen_features",
        type=bool,
        default=False,
        help="real features for seen classes",
    )
    parser.add_argument(
        "--load_embedding",
        type=str,
        default="w2c",
        choices=["attributes", "w2c", "w2c_bg", "my_w2c", "fusion", None],
    )
    parser.add_argument("--w2c_size", type=int, default=300)

    ### GENERATOR ARGS
    parser.add_argument("--noise_dim", type=int, default=300)
    parser.add_argument("--embed_dim", type=int, default=300)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--feature_dim", type=int, default=256)
    parser.add_argument("--lr_generator", type=float, default=0.0002)
    parser.add_argument("--batch_size_generator", type=int, default=128)
    parser.add_argument("--saved_validation_images", type=int, default=10)

    parser.add_argument(
        "--semantic_reconstruction",
        type=bool,
        default=False,
        help="semantic_reconstruction after feature generation",
    )
    parser.add_argument("--lbd_sr", type=float, default=0.0001)

    parser.add_argument("--context_aware", type=bool, default=True)

    args = parser.parse_args()

    voc_train = ContextSegmentation(args, split="train")

    dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample["image"].numpy()
            gt = sample["label"].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset="pascal")
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title("display")
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)


def load_obj(name):
    with open(name + ".pkl", "rb") as f:
        return pickle.load(f, encoding="latin-1")
