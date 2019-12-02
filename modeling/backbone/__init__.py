from zs3.modeling.backbone import resnet, xception, drn, mobilenet

def build_backbone(backbone, output_stride, BatchNorm, pretrained=True, imagenet_pretrained_path=''):
    if backbone == 'resnet':
        return resnet.ResNet101(output_stride, BatchNorm, pretrained=pretrained, imagenet_pretrained_path=imagenet_pretrained_path)
    else:
        raise NotImplementedError

