import os

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from zs3.dataloaders import make_data_loader
from zs3.modeling.deeplab import DeepLab
from zs3.modeling.gmmn import GMMNnetwork
from zs3.modeling.sync_batchnorm.replicate import patch_replication_callback
from zs3.utils.loss import SegmentationLosses, GMMNLoss
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
        (self.train_loader, self.val_loader, _, self.nclass,) = make_data_loader(
            args, load_embedding=args.load_embedding, w2c_size=args.w2c_size, **kwargs
        )

        model = DeepLab(
            num_classes=self.nclass,
            output_stride=args.out_stride,
            sync_bn=args.sync_bn,
            freeze_bn=args.freeze_bn,
            global_avg_pool_bn=args.global_avg_pool_bn,
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

        # Define Generator
        generator = GMMNnetwork(
            args.noise_dim, args.embed_dim, args.hidden_size, args.feature_dim
        )
        optimizer_generator = torch.optim.Adam(
            generator.parameters(), lr=args.lr_generator
        )

        class_weight = torch.ones(self.nclass)
        class_weight[args.unseen_classes_idx_metric] = args.unseen_weight
        if args.cuda:
            class_weight = class_weight.cuda()

        self.criterion = SegmentationLosses(
            weight=class_weight, cuda=args.cuda
        ).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer

        self.criterion_generator = GMMNLoss(
            sigma=[2, 5, 10, 20, 40, 80], cuda=args.cuda
        ).build_loss()
        self.generator, self.optimizer_generator = generator, optimizer_generator

        # Define Evaluator
        self.evaluator = Evaluator(
            self.nclass, args.seen_classes_idx_metric, args.unseen_classes_idx_metric
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
            self.generator = self.generator.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError(f"=> no checkpoint found at '{args.resume}'")
            checkpoint = torch.load(args.resume)
            # args.start_epoch = checkpoint['epoch']

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

            # self.best_pred = checkpoint['best_pred']
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def training(self, epoch, args):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            if len(sample["image"]) > 1:
                image, target, embedding = (
                    sample["image"],
                    sample["label"],
                    sample["label_emb"],
                )
                if self.args.cuda:
                    image, target, embedding = (
                        image.cuda(),
                        target.cuda(),
                        embedding.cuda(),
                    )
                self.scheduler(self.optimizer, i, epoch, self.best_pred)
                # ===================real feature extraction=====================
                with torch.no_grad():
                    real_features = self.model.module.forward_before_class_prediction(
                        image
                    )

                # ===================fake feature generation=====================
                fake_features = torch.zeros(real_features.shape)
                if args.cuda:
                    fake_features = fake_features.cuda()
                generator_loss_batch = 0.0
                for (
                    count_sample_i,
                    (real_features_i, target_i, embedding_i),
                ) in enumerate(zip(real_features, target, embedding)):
                    generator_loss_sample = 0.0
                    ## reduce to real feature size
                    real_features_i = (
                        real_features_i.permute(1, 2, 0)
                        .contiguous()
                        .view((-1, args.feature_dim))
                    )
                    target_i = nn.functional.interpolate(
                        target_i.view(1, 1, target_i.shape[0], target_i.shape[1]),
                        size=(real_features.shape[2], real_features.shape[3]),
                        mode="nearest",
                    ).view(-1)
                    embedding_i = nn.functional.interpolate(
                        embedding_i.view(
                            1,
                            embedding_i.shape[0],
                            embedding_i.shape[1],
                            embedding_i.shape[2],
                        ),
                        size=(real_features.shape[2], real_features.shape[3]),
                        mode="nearest",
                    )

                    embedding_i = (
                        embedding_i.permute(0, 2, 3, 1)
                        .contiguous()
                        .view((-1, args.embed_dim))
                    )

                    fake_features_i = torch.zeros(real_features_i.shape)
                    if args.cuda:
                        fake_features_i = fake_features_i.cuda()

                    unique_class = torch.unique(target_i)

                    ## test if image has unseen class pixel, if yes means no training for generator and generated features for the whole image
                    has_unseen_class = False
                    for u_class in unique_class:
                        if u_class in args.unseen_classes_idx_metric:
                            has_unseen_class = True

                    for idx_in in unique_class:
                        if idx_in != 255:
                            self.optimizer_generator.zero_grad()
                            idx_class = target_i == idx_in
                            real_features_class = real_features_i[idx_class]
                            embedding_class = embedding_i[idx_class]

                            z = torch.rand((embedding_class.shape[0], args.noise_dim))
                            if args.cuda:
                                z = z.cuda()

                            fake_features_class = self.generator(
                                embedding_class, z.float()
                            )

                            if (
                                idx_in in args.seen_classes_idx_metric
                                and not has_unseen_class
                            ):
                                ## in order to avoid CUDA out of memory
                                random_idx = torch.randint(
                                    low=0,
                                    high=fake_features_class.shape[0],
                                    size=(args.batch_size_generator,),
                                )
                                g_loss = self.criterion_generator(
                                    fake_features_class[random_idx],
                                    real_features_class[random_idx],
                                )
                                generator_loss_sample += g_loss.item()
                                g_loss.backward()
                                self.optimizer_generator.step()

                            fake_features_i[idx_class] = fake_features_class.clone()
                    generator_loss_batch += generator_loss_sample / len(unique_class)
                    if args.real_seen_features and not has_unseen_class:
                        fake_features[count_sample_i] = real_features_i.view(
                            (
                                fake_features.shape[2],
                                fake_features.shape[3],
                                args.feature_dim,
                            )
                        ).permute(2, 0, 1)
                    else:
                        fake_features[count_sample_i] = fake_features_i.view(
                            (
                                fake_features.shape[2],
                                fake_features.shape[3],
                                args.feature_dim,
                            )
                        ).permute(2, 0, 1)
                # ===================classification=====================
                self.optimizer.zero_grad()
                output = self.model.module.forward_class_prediction(
                    fake_features.detach(), image.size()[2:]
                )
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                # ===================log=====================
                tbar.set_description(
                    f" G loss: {generator_loss_batch:.3f}"
                    + " C loss: %.3f" % (train_loss / (i + 1))
                )
                self.writer.add_scalar(
                    "train/total_loss_iter", loss.item(), i + num_img_tr * epoch
                )
                self.writer.add_scalar(
                    "train/generator_loss", generator_loss_batch, i + num_img_tr * epoch
                )

                # Show 10 * 3 inference results each epoch
                if i % (num_img_tr // 10) == 0:
                    global_step = i + num_img_tr * epoch
                    self.summary.visualize_image(
                        self.writer,
                        self.args.dataset,
                        image,
                        target,
                        output,
                        global_step,
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

    def validation(self, epoch, args):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc="\r")
        test_loss = 0.0

        saved_images = {}
        saved_target = {}
        saved_prediction = {}
        for idx_unseen_class in args.unseen_classes_idx_metric:
            saved_images[idx_unseen_class] = []
            saved_target[idx_unseen_class] = []
            saved_prediction[idx_unseen_class] = []

        for i, sample in enumerate(tbar):
            image, target, embedding = (
                sample["image"],
                sample["label"],
                sample["label_emb"],
            )
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description("Test loss: %.3f" % (test_loss / (i + 1)))
            ## save image for tensorboard
            for idx_unseen_class in args.unseen_classes_idx_metric:
                if len((target.reshape(-1) == idx_unseen_class).nonzero()) > 0:
                    if len(saved_images[idx_unseen_class]) < args.saved_validation_images:
                        saved_images[idx_unseen_class].append(image.clone().cpu())
                        saved_target[idx_unseen_class].append(target.clone().cpu())
                        saved_prediction[idx_unseen_class].append(output.clone().cpu())

            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

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
        print(f"Overall: Acc:{Acc}, Acc_class:{Acc_class}, mIoU:{mIoU}, fwIoU: {FWIoU}")
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
            CLASSES_NAMES, Acc_class_by_class, mIoU_by_class
        ):
            self.writer.add_scalar("Acc_by_class/" + class_name, acc_value, epoch)
            self.writer.add_scalar("mIoU_by_class/" + class_name, mIoU_value, epoch)
            print(class_name, "- acc:", acc_value, " mIoU:", mIoU_value)

        new_pred = mIoU_unseen

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
            generator_state={
                "epoch": epoch + 1,
                "state_dict": self.generator.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best_pred": self.best_pred,
            },
        )

        global_step = epoch + 1
        for idx_unseen_class in args.unseen_classes_idx_metric:
            if len(saved_images[idx_unseen_class]) > 0:
                nb_image = len(saved_images[idx_unseen_class])
                if nb_image > args.saved_validation_images:
                    nb_image = args.saved_validation_images
                for i in range(nb_image):
                    self.summary.visualize_image_validation(
                        self.writer,
                        self.args.dataset,
                        saved_images[idx_unseen_class][i],
                        saved_target[idx_unseen_class][i],
                        saved_prediction[idx_unseen_class][i],
                        global_step,
                        name="validation_"
                        + CLASSES_NAMES[idx_unseen_class]
                        + "_"
                        + str(i),
                        nb_image=1,
                    )

        self.evaluator.reset()


def main():
    parser = get_parser()
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
        default=20,
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
    # checking point

    parser.add_argument(
        "--imagenet_pretrained_path",
        type=str,
        default="checkpoint/resnet_backbone_pretrained_imagenet_wo_pascalcontext.pth.tar",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default="checkpoint/deeplab_pretrained_pascal_context_02_unseen.pth.tar",
        help="put the path to resuming file if needed",
    )

    parser.add_argument(
        "--checkname",
        type=str,
        default="gmmn_context_w2c300_linear_weighted100_hs256_2_unseen",
    )

    # false if embedding resume
    parser.add_argument("--global_avg_pool_bn", type=bool, default=True)

    # evaluation option
    parser.add_argument(
        "--eval-interval", type=int, default=1, help="evaluation interval (default: 1)"
    )

    # keep empty
    parser.add_argument("--unseen_classes_idx", type=int, default=[])

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

    unseen_classes_idx_metric = []
    for name in unseen_names:
        unseen_classes_idx_metric.append(CLASSES_NAMES.index(name))

    ### FOR METRIC COMPUTATION IN ORDER TO GET PERFORMANCES FOR TWO SETS
    seen_classes_idx_metric = np.arange(60)

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
        "--random_last_layer", type=bool, default=True, help="randomly init last layer"
    )

    parser.add_argument(
        "--real_seen_features",
        type=bool,
        default=True,
        help="real features for seen classes",
    )
    parser.add_argument(
        "--load_embedding",
        type=str,
        default="my_w2c",
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
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):

        trainer.training(epoch, args)
        if not trainer.args.no_val and epoch % args.eval_interval == (
            args.eval_interval - 1
        ):
            trainer.validation(epoch, args)

    trainer.writer.close()


if __name__ == "__main__":
    main()
