from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from ZS3.mypath import Path
from torchvision import transforms
from ZS3.dataloaders import custom_transforms as tr
import pickle
import torch



class VOCSegmentation(Dataset):
    class_names = [
        'background',  # class 0
        'aeroplane',  # class 1
        'bicycle',  # class 2
        'bird',  # class 3
        'boat',  # class 4
        'bottle',  # class 5
        'bus',  # class 6
        'car',  # class 7
        'cat',  # class 8
        'chair',  # class 9
        'cow',  # class 10
        'diningtable',  # class 11
        'dog',  # class 12
        'horse',  # class 13
        'motorbike',  # class 14
        'person',  # class 15
        'potted plant',  # class 16
        'sheep',  # class 17
        'sofa',  # class 18
        'train',  # class 19
        'tv/monitor',  # class 20
    ]
    """
    PascalVoc dataset
    """
    NUM_CLASSES = 21

    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('pascal'),
                 split='train',
                 load_embedding=None,
                 w2c_size=300,
                 weak_label=False,
                 unseen_classes_idx_weak=[],
                 transform=True
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'JPEGImages')
        self._cat_dir = os.path.join(self._base_dir, 'SegmentationClass')

        self.transform = transform
        self.weak_label = weak_label
        self.unseen_classes_idx_weak = unseen_classes_idx_weak

        self.split = split

        self.args = args

        self.load_embedding = load_embedding
        self.w2c_size = w2c_size
        if self.load_embedding:
            self.init_embeddings()


        _splits_dir = os.path.join(self._base_dir, 'ImageSets', 'Segmentation')

        self.im_ids = []
        self.images = []
        self.categories = []


        with open(os.path.join(os.path.join(_splits_dir, self.split + '.txt')), "r") as f:
            lines = f.read().splitlines()

        for ii, line in enumerate(lines):
            _image = os.path.join(self._image_dir, line + ".jpg")
            _cat = os.path.join(self._cat_dir, line + ".png")
            assert os.path.isfile(_image)
            assert os.path.isfile(_cat)

            # if unseen classes and training split
            if len(args.unseen_classes_idx) > 0 and self.split == 'train':
                cat = Image.open(_cat)
                cat = np.array(cat, dtype=np.uint8)
                if self.lbl_contains_unseen(cat, args.unseen_classes_idx):
                    continue

            self.im_ids.append(line)
            self.images.append(_image)
            self.categories.append(_cat)

        assert (len(self.images) == len(self.categories))

        # Display stats
        print('(pascal) Number of images in {}: {:d}'.format(split, len(self.images)))


    def lbl_contains_unseen(self, lbl, unseen):
        unseen_pixel_mask = np.in1d(lbl.ravel(), unseen)
        if np.sum(unseen_pixel_mask) > 0: # ignore images with any train_unseen pixels
            return True
        return False


    def init_embeddings(self):
        embed_arr = load_obj('embeddings/pascal/w2c/norm_embed_arr_' + str(self.w2c_size))
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
                _target = Image.open('weak_label_pascal_10_unseen_top_by_image_25.0/pascal/'+(self.categories[index].split('/'))[-1].split('.')[0]+'.jpg')

        sample = {'image': _img, 'label': _target}

        if self.transform:
            if self.split == 'train':
                sample = self.transform_tr(sample)
            elif self.split == 'val':
                sample = self.transform_val(sample)
        else:
            sample = self.transform_weak(sample)


        if self.load_embedding:
            mask = (sample['label'] == 255)
            sample['label'][mask] = 0
            lbl_vec = self.embeddings(sample['label'].long()).data
            lbl_vec = lbl_vec.permute(2, 0, 1)
            sample['label'][mask] = 255
            sample['label_emb'] = lbl_vec
            #sample = {'image': sample['image'], 'label': sample['label'], 'label_emb': lbl_vec}
        sample['image_name'] = self.images[index]
        return sample


    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.categories[index])
        return _img, _target

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size, fill=255),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.FixScale(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])


        return composed_transforms(sample)

    def transform_weak(self, sample):

        composed_transforms = transforms.Compose([
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def __str__(self):
        return 'VOC2012(split=' + str(self.split) + ')'


if __name__ == '__main__':
    from ZS3.dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    voc_train = VOCSegmentation(args, split='train')

    dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='pascal')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f, encoding='latin-1')