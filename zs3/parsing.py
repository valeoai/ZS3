import argparse


def get_parser():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument(
        "--workers", type=int, default=6, metavar="N", help="dataloader threads"
    )
    parser.add_argument(
        "--freeze-bn",
        type=bool,
        default=False,
        help="whether to freeze bn parameters (default: False)",
    )
    return parser
