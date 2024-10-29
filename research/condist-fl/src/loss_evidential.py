#%%
from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple, Union

import torch
from monai.losses import DiceCELoss, MaskedDiceLoss
from monai.networks import one_hot
from monai.utils import LossReduction
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss
from nvflare.app_common.app_constant import AppConstants

from monai.utils import pytorch_after


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
        softmax: bool = False,
        kl_weight: float = 1e-2,
        annealing_step: int = 10
    ):
        super().__init__()

        self.transform = MarginalTransform(foreground, softmax=softmax)
        self.kl_weight = kl_weight
        self.annealing_step = annealing_step
        self.smooth = 1e-5
        
    def kl_divergence(self, alpha):
        shape = list(alpha.shape)
        shape[0] = 1
        ones = torch.ones(tuple(shape)).cuda()

        S = torch.sum(alpha, dim=1, keepdim=True) 

        first_term = (torch.lgamma(S)
                      - torch.lgamma(alpha).sum(dim=1, keepdim=True)
                      - torch.lgamma(ones.sum(dim=1, keepdim=True))
                      )
        
        second_term = ((alpha - ones).mul(torch.digamma(alpha) - torch.digamma(S)).sum(dim=1, keepdim=True))
        kl = first_term + second_term

        return kl.mean() 
    

    def forward(self, logits: Tensor, targets: Tensor, current_round: int):
        C = logits.shape[1]
        logits, targets = self.transform(logits, targets)
        evidence = F.relu(logits)
        alpha = (evidence + 1)**2
        S = torch.sum(alpha, dim=1, keepdim=True)
        rho = alpha / S

        # Task loss (Evidential Dice Loss)
        dice_score = 0
        for class_idx in range(logits.shape[1]):
            inter = (rho[:,class_idx,...] * targets[:,class_idx,...]).sum()
            union = (rho[:,class_idx,...] ** 2).sum() + (targets[:,class_idx,...] ** 2).sum() + (rho[:,class_idx,...]*(1-rho[:,class_idx,...])/(S[:,0,...]+1)).sum()
            dice_score += (2*inter + self.smooth) / (union + self.smooth)
        loss_task = 1 - dice_score/logits.shape[1]

        # Regularization loss
        anneling_coef = torch.min(
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(current_round / self.annealing_step, dtype=torch.float)
        )

        # KL divergence with uniform prior
        kl_alpha = targets + (1 - targets) * alpha
        loss_kl = anneling_coef*self.kl_divergence(kl_alpha)
        loss_cor = torch.sum(- C/S.detach() * targets * logits) / (logits.shape[0] * logits.shape[2] * logits.shape[3])
        loss_reg = loss_kl + loss_cor

        # Total loss
        loss = loss_task + self.kl_weight * loss_reg

        return loss
    

class AdaptiveDeepSupervisionLoss(_Loss):
    """
    Wrapper class around the main loss function to accept a list of tensors returned from a deeply
    supervised networks. The final loss is computed as the sum of weighted losses for each of deep supervision levels.
    """

    def __init__(self, loss: _Loss, weight_mode: str = "exp", weights: list[float] | None = None) -> None:
        """
        Args:
            loss: main loss instance, e.g DiceLoss().
            weight_mode: {``"same"``, ``"exp"``, ``"two"``}
                Specifies the weights calculation for each image level. Defaults to ``"exp"``.
                - ``"same"``: all weights are equal to 1.
                - ``"exp"``: exponentially decreasing weights by a power of 2: 0, 0.5, 0.25, 0.125, etc .
                - ``"two"``: equal smaller weights for lower levels: 1, 0.5, 0.5, 0.5, 0.5, etc
            weights: a list of weights to apply to each deeply supervised sub-loss, if provided, this will be used
                regardless of the weight_mode
        """
        super().__init__()
        self.loss = loss
        self.weight_mode = weight_mode
        self.weights = weights
        self.interp_mode = "nearest-exact" if pytorch_after(1, 11) else "nearest"

    def get_weights(self, levels: int = 1) -> list[float]:
        """
        Calculates weights for a given number of scale levels
        """
        levels = max(1, levels)
        if self.weights is not None and len(self.weights) >= levels:
            weights = self.weights[:levels]
        elif self.weight_mode == "same":
            weights = [1.0] * levels
        elif self.weight_mode == "exp":
            weights = [max(0.5**l, 0.0625) for l in range(levels)]
        elif self.weight_mode == "two":
            weights = [1.0 if l == 0 else 0.5 for l in range(levels)]
        else:
            weights = [1.0] * levels

        return weights

    def get_loss(self, input: torch.Tensor, target: torch.Tensor, current_round: int) -> torch.Tensor:
        """
        Calculates a loss output accounting for differences in shapes,
        and downsizing targets if necessary (using nearest neighbor interpolation)
        Generally downsizing occurs for all level, except for the first (level==0)
        """
        if input.shape[2:] != target.shape[2:]:
            target = F.interpolate(target, size=input.shape[2:], mode=self.interp_mode)
        return self.loss(input, target, current_round)  # type: ignore[no-any-return]

    def forward(self, input: Union[None, torch.Tensor, list[torch.Tensor]], target: torch.Tensor, current_round: int) -> torch.Tensor:
        if isinstance(input, (list, tuple)):
            weights = self.get_weights(levels=len(input))
            loss = torch.tensor(0, dtype=torch.float, device=target.device)
            for l in range(len(input)):
                loss += weights[l] * self.get_loss(input[l].float(), target, current_round)
            return loss
        if input is None:
            raise ValueError("input shouldn't be None.")

        return self.loss(input.float(), target, current_round)  # type: ignore[no-any-return]


ds_loss = AdaptiveDeepSupervisionLoss


#%%

# import torch

# preds = torch.randn(4, 8, 128, 128, 128)
# targets = torch.randint(0, 8, (4, 1, 128, 128, 128))
# current_round = 2

# loss_fn = MarginalEvidentialLoss(foreground=[1, 2])
# loss = loss_fn(preds, targets, current_round)

# print("loss:", loss.item())

#%%