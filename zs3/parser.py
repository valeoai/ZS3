import argparse


def get_parser():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
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

    parser.add_argument(
        "--workers", type=int, default=4, metavar="N", help="dataloader threads"
    )
    parser.add_argument("--base-size", type=int, default=312, help="base image size")
    parser.add_argument("--crop-size", type=int, default=312, help="crop image size")
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
        default=200,
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
        default=10,
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

    parser.add_argument(
        "--exp_path", type=str, default="run", help="set the checkpoint name"
    )

    # finetuning pre-trained models
    parser.add_argument(
        "--ft",
        action="store_true",
        default=False,
        help="finetuning on a different dataset",
    )
    # evaluation option
    parser.add_argument(
        "--eval-interval", type=int, default=10, help="evaluation interval (default: 1)"
    )
    parser.add_argument(
        "--no-val",
        action="store_true",
        default=False,
        help="skip validation during training",
    )
    return parser