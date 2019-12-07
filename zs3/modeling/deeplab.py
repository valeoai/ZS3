import torch
import torch.nn as nn
import torch.nn.functional as F

from zs3.modeling.aspp import build_aspp
from zs3.modeling.backbone import build_backbone
from zs3.modeling.decoder import build_decoder
from zs3.modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class DeepLabNonLinearClassifier(nn.Module):
    def __init__(self, args, num_classes, global_avg_pool_bn=True):
        super().__init__()

        self.deeplab = DeepLab(
            num_classes=num_classes,
            backbone=args.backbone,
            output_stride=args.out_stride,
            sync_bn=args.sync_bn,
            freeze_bn=args.freeze_bn,
            global_avg_pool_bn=global_avg_pool_bn,
        )

        self.nonlinear_conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1), nn.ReLU(), nn.Dropout(0.1)
        )
        self.pred_conv = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)

    def forward(self, x, input_size):
        x = self.deeplab.forward_before_class_prediction(x)
        x = self.nonlinear_conv(x)
        x = self.pred_conv(x)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=True)
        return x

    def forward_before_class_prediction(self, x):
        x = self.deeplab.forward_before_class_prediction(x)
        return x

    def forward_class_prediction(self, x, input_size):
        x = self.nonlinear_conv(x)
        x = self.pred_conv(x)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=True)
        return x


class DeepLab(nn.Module):
    def __init__(
        self,
        backbone="resnet",
        output_stride=16,
        num_classes=21,
        sync_bn=True,
        freeze_bn=False,
        pretrained=True,
        global_avg_pool_bn=True,
        imagenet_pretrained_path="",
    ):
        super().__init__()
        if backbone == "drn":
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(
            backbone,
            output_stride,
            BatchNorm,
            pretrained=pretrained,
            imagenet_pretrained_path=imagenet_pretrained_path,
        )
        self.aspp = build_aspp(backbone, output_stride, BatchNorm, global_avg_pool_bn)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

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

    def forward_class_prediction_1d(self, x):
        x = self.decoder.forward_class_prediction(x)
        return x

    def forward_before_last_conv_finetune(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder.forward_before_last_conv_finetune(x, low_level_feat)
        return x

    def forward_class_last_conv_finetune(self, x):
        x = self.decoder.forward_class_last_conv_finetune(x)
        return x

    def forward_before_decoder(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        return x, low_level_feat

    def forward_decoder(self, x, low_level_feat, input_size):
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=True)
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


class DeepLabEmbedding(nn.Module):
    def __init__(
        self,
        backbone="resnet",
        output_stride=16,
        embed_dim=300,
        sync_bn=True,
        freeze_bn=False,
        pretrained=True,
        global_avg_pool_bn=False,
    ):
        super().__init__()
        if backbone == "drn":
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d
        self.backbone = build_backbone(
            backbone, output_stride, BatchNorm, pretrained=pretrained
        )
        self.aspp = build_aspp(backbone, output_stride, BatchNorm, global_avg_pool_bn)
        self.decoder = build_decoder(embed_dim, backbone, BatchNorm)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode="bilinear", align_corners=True)
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


if __name__ == "__main__":
    model = DeepLab(backbone="mobilenet", output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())
