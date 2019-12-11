from zs3.modeling.backbone import resnet


def build_backbone(
    output_stride, BatchNorm, pretrained=True, imagenet_pretrained_path=""
):
    return resnet.ResNet101(
        output_stride,
        BatchNorm,
        pretrained=pretrained,
        imagenet_pretrained_path=imagenet_pretrained_path,
    )
