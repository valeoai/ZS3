import os

import numpy as np
import torch
from tqdm import tqdm

from zs3.dataloaders import make_data_loader
from zs3.modeling.deeplab import DeepLab
from zs3.modeling.sync_batchnorm.replicate import patch_replication_callback
from zs3.mypath import Path
from zs3.utils.calculate_weights import calculate_weigths_labels
from zs3.utils.loss import SegmentationLosses
from zs3.utils.lr_scheduler import LR_Scheduler
from zs3.utils.metrics import Evaluator
from zs3.utils.saver import Saver
from zs3.utils.summaries import TensorboardSummary
from zs3.parsing import get_parser
from zs3.exp_data import CLASSES_NAMES


class Trainer:
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        # Define Dataloader
        kwargs = {"num_workers": args.workers, "pin_memory": True}
        (
            self.train_loader,
            self.val_loader,
            _,
            self.nclass,
        ) = make_data_loader(args, **kwargs)

        # Define network
        model = DeepLab(
            num_classes=self.nclass,
            output_stride=args.out_stride,
            sync_bn=args.sync_bn,
            freeze_bn=args.freeze_bn,
            pretrained=args.imagenet_pretrained,
            imagenet_pretrained_path=args.imagenet_pretrained_path,
        )

        train_params = [
            {"params": model.get_1x_lr_params(), "lr": args.lr},
            {"params": model.get_10x_lr_params(), "lr": args.lr * 10},
        ]

        # Define Optimizer
        optimizer = torch.optim.SGD(
            train_params,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )

        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(
                Path.db_root_dir(args.dataset), args.dataset + "_classes_weights.npy"
            )
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(
                    args.dataset, self.train_loader, self.nclass
                )
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(
            mode=args.loss_type
        )
        self.model, self.optimizer = model, optimizer

        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(
            args.lr_scheduler, args.lr, args.epochs, len(self.train_loader)
        )

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError(f"=> no checkpoint found at '{args.resume}'")
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            if args.cuda:
                self.model.module.load_state_dict(checkpoint["state_dict"])
            else:
                self.model.load_state_dict(checkpoint["state_dict"])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.best_pred = checkpoint["best_pred"]
            print(
                f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})"
            )

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            if len(sample["image"]) > 1:
                image, target = sample["image"], sample["label"]
                if self.args.cuda:
                    image, target = image.cuda(), target.cuda()
                self.scheduler(self.optimizer, i, epoch, self.best_pred)
                self.optimizer.zero_grad()
                output = self.model(image)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                tbar.set_description("Train loss: %.3f" % (train_loss / (i + 1)))
                self.writer.add_scalar(
                    "train/total_loss_iter", loss.item(), i + num_img_tr * epoch
                )

                # Show 10 * 3 inference results each epoch
                if i % (num_img_tr // 10) == 0:
                    global_step = i + num_img_tr * epoch
                    self.summary.visualize_image(
                        self.writer, self.args.dataset, image, target, output, global_step
                    )

        self.writer.add_scalar("train/total_loss_epoch", train_loss, epoch)
        print(
            "[Epoch: %d, numImages: %5d]"
            % (epoch, i * self.args.batch_size + image.data.shape[0])
        )
        print(f"Loss: {train_loss:.3f}")

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": self.model.module.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "best_pred": self.best_pred,
                },
                is_best,
            )

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc="\r")
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample["image"], sample["label"]
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description("Test loss: %.3f" % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class, Acc_class_by_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU, mIoU_by_class = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar("val/total_loss_epoch", test_loss, epoch)
        self.writer.add_scalar("val/mIoU", mIoU, epoch)
        self.writer.add_scalar("val/Acc", Acc, epoch)
        self.writer.add_scalar("val/Acc_class", Acc_class, epoch)
        self.writer.add_scalar("val/fwIoU", FWIoU, epoch)
        print("Validation:")
        print(
            "[Epoch: %d, numImages: %5d]"
            % (epoch, i * self.args.batch_size + image.data.shape[0])
        )
        print(
            f"Acc:{Acc}, Acc_class:{Acc_class}, mIoU:{mIoU}, fwIoU: {FWIoU}"
        )
        print(f"Loss: {test_loss:.3f}")

        for i, (class_name, acc_value, mIoU_value) in enumerate(
            zip(CLASSES_NAMES, Acc_class_by_class, mIoU_by_class)
        ):
            self.writer.add_scalar("Acc_by_class/" + class_name, acc_value, epoch)
            self.writer.add_scalar("mIoU_by_class/" + class_name, mIoU_value, epoch)
            print(CLASSES_NAMES[i], "- acc:", acc_value, " mIoU:", mIoU_value)

        new_pred = mIoU
        is_best = True
        self.best_pred = new_pred
        self.saver.save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": self.model.module.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best_pred": self.best_pred,
            },
            is_best,
        )


def main():
    parser = get_parser()
    parser.add_argument(
        "--imagenet_pretrained",
        type=bool,
        default=True,
        help="imagenet pretrained backbone",
    )

    parser.add_argument(
        "--out-stride", type=int, default=16, help="network output stride (default: 8)"
    )

    # PASCAL VOC
    parser.add_argument(
        "--dataset",
        type=str,
        default="context",
        choices=["pascal", "coco", "cityscapes", "context"],
        help="dataset name (default: pascal)",
    )

    parser.add_argument("--base-size", type=int, default=312, help="base image size")
    parser.add_argument("--crop-size", type=int, default=312, help="crop image size")
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
        default=200,
        metavar="N",
        help="number of epochs to train (default: auto)",
    )

    # PASCAL VOC
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        metavar="N",
        help="input batch size for training (default: auto)",
    )
    # checking point
    parser.add_argument(
        "--resume", type=str, default=None, help="put the path to resuming file if needed"
    )
    parser.add_argument(
        "--checkname",
        type=str,
        default="context_2_unseen",
        help="set the checkpoint name",
    )

    parser.add_argument(
        "--imagenet_pretrained_path",
        type=str,
        default="checkpoint/resnet_backbone_pretrained_imagenet_wo_pascalcontext.pth.tar",
        help="set the checkpoint name",
    )

    # evaluation option
    parser.add_argument(
        "--eval-interval", type=int, default=10, help="evaluation interval (default: 1)"
    )

    # 2 unseen
    unseen_names = ["cow", "motorbike"]
    # 4 unseen
    # unseen_names = ['cow', 'motorbike', 'sofa', 'cat']
    # 6 unseen
    # unseen_names = ['cow', 'motorbike', 'sofa', 'cat', 'boat', 'fence']
    # 8 unseen
    # unseen_names = ['cow', 'motorbike', 'sofa', 'cat', 'boat', 'fence', 'bird', 'tvmonitor']
    # 10 unseen
    # unseen_names = ['cow', 'motorbike', 'sofa', 'cat', 'boat', 'fence', 'bird', 'tvmonitor', 'aeroplane', 'keyboard']

    unseen_classes_idx = []
    for name in unseen_names:
        unseen_classes_idx.append(CLASSES_NAMES.index(name))
    print(unseen_classes_idx)
    # all classes
    parser.add_argument("--unseen_classes_idx", type=int, default=unseen_classes_idx)
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(",")]
        except ValueError:
            raise ValueError(
                "Argument --gpu_ids must be a comma-separated list of integers only"
            )

    args.sync_bn = args.cuda and len(args.gpu_ids) > 1

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            "coco": 30,
            "cityscapes": 200,
            "pascal": 50,
            "pascal": 150,
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            "coco": 0.1,
            "cityscapes": 0.01,
            "pascal": 0.007,
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size

    if args.checkname is None:
        args.checkname = "deeplab-resnet"
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print("Starting Epoch:", trainer.args.start_epoch)
    print("Total Epoches:", trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (
            args.eval_interval - 1
        ):
            trainer.validation(epoch)
    trainer.writer.close()


if __name__ == "__main__":
    main()
