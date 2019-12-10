import os

import numpy as np
import torch
from tqdm import tqdm

from zs3.dataloaders import make_data_loader
from zs3.modeling.deeplab import DeepLab
from zs3.modeling.sync_batchnorm.replicate import patch_replication_callback
from zs3.dataloaders.datasets import DATASETS_DIRS
from zs3.utils.calculate_weights import calculate_weigths_labels
from zs3.utils.loss import SegmentationLosses
from zs3.utils.lr_scheduler import LR_Scheduler
from zs3.utils.metrics import Evaluator, Evaluator_seen_unseen
from zs3.utils.saver import Saver
from zs3.utils.summaries import TensorboardSummary
from zs3.parsing import get_parser


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

        model = DeepLab(
            num_classes=self.nclass,
            output_stride=args.out_stride,
            sync_bn=args.sync_bn,
            freeze_bn=args.freeze_bn,
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
            classes_weights_path = DATASETS_DIRS[args.dataset] / args.dataset + "_classes_weights.npy"

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
        self.evaluator = Evaluator(
            self.nclass, args.seen_classes_idx_metric, args.unseen_classes_idx_metric
        )
        self.evaluator_seen_unseen = Evaluator_seen_unseen(
            self.nclass, args.unseen_classes_idx_metric
        )

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

            if args.random_last_layer:
                checkpoint["state_dict"]["decoder.pred_conv.weight"] = torch.rand(
                    (
                        self.nclass,
                        checkpoint["state_dict"]["decoder.pred_conv.weight"].shape[1],
                        checkpoint["state_dict"]["decoder.pred_conv.weight"].shape[2],
                        checkpoint["state_dict"]["decoder.pred_conv.weight"].shape[3],
                    )
                )
                checkpoint["state_dict"]["decoder.pred_conv.bias"] = torch.rand(
                    self.nclass
                )

            if args.cuda:
                self.model.module.load_state_dict(checkpoint["state_dict"])
            else:
                self.model.load_state_dict(checkpoint["state_dict"])

            if not args.ft:
                if not args.nonlinear_last_layer:
                    self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.best_pred = checkpoint["best_pred"]
            print(
                f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})"
            )

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def validation(self, epoch, args):
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
            "diningtable",  # class 11
            "dog",  # class 12
            "horse",  # class 13
            "motorbike",  # class 14
            "person",  # class 15
            "potted plant",  # class 16
            "sheep",  # class 17
            "sofa",  # class 18
            "train",  # class 19
            "tv/monitor",  # class 20
        ]
        self.model.eval()
        self.evaluator.reset()
        all_target = []
        all_pred = []
        all_pred_unseen = []
        tbar = tqdm(self.val_loader, desc="\r")
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample["image"], sample["label"]
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                if args.nonlinear_last_layer:
                    output = self.model(image, image.size()[2:])
                else:
                    output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description("Test loss: %.3f" % (test_loss / (i + 1)))
            pred = output.data.cpu().numpy()
            pred_unseen = pred.copy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)

            pred_unseen[:, args.seen_classes_idx_metric] = float("-inf")
            pred_unseen = np.argmax(pred_unseen, axis=1)

            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

            all_target.append(target)
            all_pred.append(pred)
            all_pred_unseen.append(pred_unseen)

        # Fast test during the training
        Acc, Acc_seen, Acc_unseen = self.evaluator.Pixel_Accuracy()
        (
            Acc_class,
            Acc_class_by_class,
            Acc_class_seen,
            Acc_class_unseen,
        ) = self.evaluator.Pixel_Accuracy_Class()
        (
            mIoU,
            mIoU_by_class,
            mIoU_seen,
            mIoU_unseen,
        ) = self.evaluator.Mean_Intersection_over_Union()
        (
            FWIoU,
            FWIoU_seen,
            FWIoU_unseen,
        ) = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar("val_overall/total_loss_epoch", test_loss, epoch)
        self.writer.add_scalar("val_overall/mIoU", mIoU, epoch)
        self.writer.add_scalar("val_overall/Acc", Acc, epoch)
        self.writer.add_scalar("val_overall/Acc_class", Acc_class, epoch)
        self.writer.add_scalar("val_overall/fwIoU", FWIoU, epoch)

        self.writer.add_scalar("val_seen/mIoU", mIoU_seen, epoch)
        self.writer.add_scalar("val_seen/Acc", Acc_seen, epoch)
        self.writer.add_scalar("val_seen/Acc_class", Acc_class_seen, epoch)
        self.writer.add_scalar("val_seen/fwIoU", FWIoU_seen, epoch)

        self.writer.add_scalar("val_unseen/mIoU", mIoU_unseen, epoch)
        self.writer.add_scalar("val_unseen/Acc", Acc_unseen, epoch)
        self.writer.add_scalar("val_unseen/Acc_class", Acc_class_unseen, epoch)
        self.writer.add_scalar("val_unseen/fwIoU", FWIoU_unseen, epoch)

        print("Validation:")
        print(
            "[Epoch: %d, numImages: %5d]"
            % (epoch, i * self.args.batch_size + image.data.shape[0])
        )
        print(f"Loss: {test_loss:.3f}")
        print(
            f"Overall: Acc:{Acc}, Acc_class:{Acc_class}, mIoU:{mIoU}, fwIoU: {FWIoU}"
        )
        print(
            "Seen: Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(
                Acc_seen, Acc_class_seen, mIoU_seen, FWIoU_seen
            )
        )
        print(
            "Unseen: Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(
                Acc_unseen, Acc_class_unseen, mIoU_unseen, FWIoU_unseen
            )
        )

        for class_name, acc_value, mIoU_value in zip(
            class_names, Acc_class_by_class, mIoU_by_class
        ):
            self.writer.add_scalar("Acc_by_class/" + class_name, acc_value, epoch)
            self.writer.add_scalar("mIoU_by_class/" + class_name, mIoU_value, epoch)
            print(class_name, "- acc:", acc_value, " mIoU:", mIoU_value)

        all_target = np.concatenate(all_target)
        all_pred = np.concatenate(all_pred)
        metric_values, metric_by_class = self.evaluator_seen_unseen.label_accuracy_score(
            all_target, all_pred, by_class=True
        )
        metric_values = np.array(metric_values)
        metric_by_class = np.array(metric_by_class)

        print("pA:")
        print("overall:", metric_values[0, 0])
        print("seen:", metric_values[1, 0])
        print("unseen:", metric_values[2, 0])
        self.writer.add_scalar("val_overall/pA", metric_values[0, 0], epoch)
        self.writer.add_scalar("val_seen/pA", metric_values[1, 0], epoch)
        self.writer.add_scalar("val_unseen/pA", metric_values[2, 0], epoch)

        print("cA:")
        print("overall:", metric_values[0, 1])
        print("seen:", metric_values[1, 1])
        print("unseen:", metric_values[2, 1])
        self.writer.add_scalar("val_overall/cA", metric_values[0, 1], epoch)
        self.writer.add_scalar("val_seen/cA", metric_values[1, 1], epoch)
        self.writer.add_scalar("val_unseen/cA", metric_values[2, 1], epoch)

        print("mIoU:")
        print("overall:", metric_values[0, 2])
        print("seen:", metric_values[1, 2])
        print("unseen:", metric_values[2, 2])
        self.writer.add_scalar("val_overall/mIoU", metric_values[0, 2], epoch)
        self.writer.add_scalar("val_seen/mIoU", metric_values[1, 2], epoch)
        self.writer.add_scalar("val_unseen/mIoU", metric_values[2, 2], epoch)

        print("wIoU:")
        print("overall:", metric_values[0, 3])
        print("seen:", metric_values[1, 3])
        print("unseen:", metric_values[2, 3])
        self.writer.add_scalar("val_overall/wIoU", metric_values[0, 3], epoch)
        self.writer.add_scalar("val_seen/wIoU", metric_values[1, 3], epoch)
        self.writer.add_scalar("val_unseen/wIoU", metric_values[2, 3], epoch)

        for class_name, class_value in zip(class_names, metric_by_class):
            print(class_name, ":", class_value)

        all_pred_unseen = np.concatenate(all_pred_unseen)
        (
            metric_values_unseen,
            metric_by_class_unseen,
        ) = self.evaluator_seen_unseen.label_accuracy_score(
            all_target, all_pred_unseen, by_class=True
        )
        metric_values_unseen = np.array(metric_values_unseen)
        metric_by_class_unseen = np.array(metric_by_class_unseen)

        print("pA:")
        print("unseen:", metric_values_unseen[2, 0])
        self.writer.add_scalar("val_only_unseen/pA", metric_values_unseen[2, 0], epoch)

        print("cA:")
        print("unseen:", metric_values_unseen[2, 1])
        self.writer.add_scalar("val_only_unseen/cA", metric_values_unseen[2, 1], epoch)

        print("mIoU:")
        print("unseen:", metric_values_unseen[2, 2])
        self.writer.add_scalar("val_only_unseen/mIoU", metric_values_unseen[2, 2], epoch)

        print("wIoU:")
        print("unseen:", metric_values_unseen[2, 3])
        self.writer.add_scalar("val_only_unseen/wIoU", metric_values_unseen[2, 3], epoch)

        for class_name, class_value in zip(class_names, metric_by_class_unseen):
            print(class_name, ":", class_value)

        new_pred = mIoU
        if new_pred > self.best_pred:
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
        "--out-stride", type=int, default=16, help="network output stride (default: 8)"
    )

    # PASCAL VOC
    parser.add_argument(
        "--dataset",
        type=str,
        default="pascal",
        choices=["pascal", "coco", "cityscapes"],
        help="dataset name (default: pascal)",
    )

    parser.add_argument(
        "--use-sbd",
        action="store_true",
        default=True,
        help="whether to use SBD dataset (default: True)",
    )
    parser.add_argument("--base-size", type=int, default=513, help="base image size")
    parser.add_argument("--crop-size", type=int, default=513, help="crop image size")
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
        default=300,
        metavar="N",
        help="number of epochs to train (default: auto)",
    )

    # PASCAL VOC
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        metavar="N",
        help="input batch size for training (default: auto)",
    )
    # cuda, seed and logging
    # checking point
    parser.add_argument(
        "--imagenet_pretrained_path",
        type=str,
        default="checkpoint/resnet_backbone_pretrained_imagenet_wo_pascalvoc.pth.tar",
    )
    # checking point
    parser.add_argument(
        "--resume",
        type=str,
        default="checkpoint/deeplab_pascal_voc_02_unseen_GMMN_final.pth.tar",
        help="put the path to resuming file if needed",
    )

    parser.add_argument("--checkname", type=str, default="pascal_eval")

    # evaluation option
    parser.add_argument(
        "--eval-interval", type=int, default=5, help="evaluation interval (default: 1)"
    )
    ### FOR IMAGE SELECTION IN ORDER TO TAKE OFF IMAGE WITH UNSEEN CLASSES FOR TRAINING AND VALIDATION
    # keep empty
    parser.add_argument("--unseen_classes_idx", type=int, default=[])

    ### FOR METRIC COMPUTATION IN ORDER TO GET PERFORMANCES FOR TWO SETS
    seen_classes_idx_metric = np.arange(21)

    # 2 unseen
    unseen_classes_idx_metric = [10, 14]
    # 4 unseen
    # unseen_classes_idx_metric = [10, 14, 1, 18]
    # 6 unseen
    # unseen_classes_idx_metric = [10, 14, 1, 18, 8, 20]
    # 8 unseen
    # unseen_classes_idx_metric = [10, 14, 1, 18, 8, 20, 19, 5]
    # 10 unseen
    # unseen_classes_idx_metric = [10, 14, 1, 18, 8, 20, 19, 5, 9, 16]

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
        "--nonlinear_last_layer", type=bool, default=False, help="non linear prediction"
    )
    parser.add_argument(
        "--random_last_layer", type=bool, default=False, help="randomly init last layer"
    )

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
    trainer.validation(0, args)


if __name__ == "__main__":
    main()
