import torch.nn as nn
import torch.nn.functional as F

from zs3.modeling.aspp import build_aspp
from zs3.modeling.backbone import build_backbone
from zs3.modeling.decoder import build_decoder
from zs3.modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class DeepLab(nn.Module):
    def __init__(
        self,
        output_stride=16,
        num_classes=21,
        sync_bn=True,
        freeze_bn=False,
        pretrained=True,
        global_avg_pool_bn=True,
        imagenet_pretrained_path="",
    ):
        super().__init__()

        if sync_bn:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(
            output_stride,
            BatchNorm,
            pretrained=pretrained,
            imagenet_pretrained_path=imagenet_pretrained_path,
        )
        self.aspp = build_aspp(output_stride, BatchNorm, global_avg_pool_bn)
        self.decoder = build_decoder(num_classes, BatchNorm)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode="bilinear", align_corners=True)
        return x

    def forward_before_class_prediction(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder.forward_before_class_prediction(x, low_level_feat)
        return x

    def forward_class_prediction(self, x, input_size):
        x = self.decoder.forward_class_prediction(x)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=True)
        return x

    def forward_before_last_conv_finetune(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder.forward_before_last_conv_finetune(x, low_level_feat)
        return x

    def forward_class_last_conv_finetune(self, x):
        x = self.decoder.forward_class_last_conv_finetune(x)
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if (
                    isinstance(m[1], nn.Conv2d)
                    or isinstance(m[1], SynchronizedBatchNorm2d)
                    or isinstance(m[1], nn.BatchNorm2d)
                ):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if (
                    isinstance(m[1], nn.Conv2d)
                    or isinstance(m[1], SynchronizedBatchNorm2d)
                    or isinstance(m[1], nn.BatchNorm2d)
                ):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p
