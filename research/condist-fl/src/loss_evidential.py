#%%

from typing import Callable, Optional, Sequence, Tuple, Union

import torch
from monai.losses import DiceCELoss, MaskedDiceLoss
from monai.networks import one_hot
from monai.utils import LossReduction
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss


class MarginalTransform(object):
    def __init__(self, foreground: Sequence[int], softmax: bool = False):
        self.foreground = foreground
        self.softmax = softmax

    def reduce_background_channels(self, tensor: Tensor) -> Tensor:
        n_chs = tensor.shape[1]
        slices = torch.split(tensor, 1, dim=1)

        fg = [slices[i] for i in self.foreground]
        bg = sum([slices[i] for i in range(n_chs) if i not in self.foreground])

        output = torch.cat([bg] + fg, dim=1)
        return output

    def __call__(self, preds: Tensor, target: Tensor) -> Tuple[Tensor]:
        n_pred_ch = preds.shape[1]
        if n_pred_ch == 1:
            # Marginal loss is not intended for single channel output
            return preds, target

        if self.softmax:
            preds = torch.softmax(preds, 1)

        if target.shape[1] == 1:
            target = one_hot(target, num_classes=n_pred_ch)
        elif target.shape != n_pred_ch:
            raise ValueError(f"Number of channels of label must be 1 or {n_pred_ch}.")

        preds = self.reduce_background_channels(preds)
        target = self.reduce_background_channels(target)

        return preds, target


class MarginalEvidentialLoss(_Loss):
    def __init__(
        self,
        foreground: Sequence[int],
        include_background: bool = True,
        softmax: bool = False,
        other_act: Optional[Callable] = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: str = "mean",
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        ce_weight: Optional[Tensor] = None,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
    ):
        super().__init__()

        self.transform = MarginalTransform(foreground, softmax=softmax)
        

    def forward(self, preds: Tensor, targets: Tensor):
        preds, targets = self.transform(preds, targets)
        evidence = F.relu(preds)
        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        rho = alpha / S
        dims = tuple(range(2, preds.ndim))

        numerator = (rho * targets).sum(dim=dims)
        denom1 = (targets ** 2).sum(dim=dims)
        denom2 = (rho ** 2).sum(dim=dims)
        denom3 = ((rho * (1 - rho)) / S).sum(dim=dims)

        dice = torch.mean(2 * numerator / (denom1 + denom2 + denom3 + 1e-6), dim=1)
        loss = (1 - dice).mean()

        return loss

#%%

# import torch

# preds = torch.randn(4, 8, 128, 128, 128)
# targets = torch.randint(0, 8, (4, 1, 128, 128, 128))
# loss_fn = MarginalEvidentialLoss(foreground=[1, 2, 3, 4, 5, 6, 7], softmax=True)
# loss = loss_fn(preds, targets)
# loss









#%%