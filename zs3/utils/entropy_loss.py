import numpy as np
import torch
import torch.nn as nn


class EntropyLoss(nn.Module):
    def __init__(self, weight=None, hardneg=False, hardnegPer=-1):
        super().__init__()
        self.weight = weight
        self.hardneg = hardneg
        if self.hardneg:
            assert hardnegPer > 0, "Hardneg percentage must be between 0 and 1."
        self.hardnegPer = hardnegPer

    def forward(self, v):
        """
            Args:
                predict: probabilistic prediction map (n, c, h, w)
        """
        assert v.dim() == 4
        n, c, h, w = v.size()
        if self.weight is None:
            if not self.hardneg:
                loss = -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (
                    n * h * w * np.log2(c)
                )
            else:
                entp = -torch.sum(torch.mul(v, torch.log2(v + 1e-30)), dim=1) / np.log2(c)
                thres = torch.topk(entp.reshape((-1,)), int(h * w * self.hardnegPer))[
                    -1
                ]  # determine kth largest threshold
                mask = (entp > thres).float() * 1
                entp_masked = entp * mask
                loss = torch.sum(entp_masked) / (h * w)
        else:
            loss = -torch.sum(
                torch.mul(torch.mul(v, self.weight), torch.log2(v + 1e-30))
            ) / (h * w * np.log2(c))

        return loss


class EntropyLossOnlySeen(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, v):
        """
            Args:
                predict: probabilistic prediction map (n, c)
        """
        assert v.dim() == 2
        n, c = v.size()
        loss = -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * np.log2(c))

        return loss
