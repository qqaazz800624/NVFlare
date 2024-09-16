from typing import Any, Dict

import torch
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceLoss, DiceCELoss
from monai.transforms import AsDiscreted
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import autocast


def get_fg_classes(fg_idx, classes):
    out = {}
    for idx in fg_idx:
        out[classes[idx]] = idx
    return out


class Validator(object):
    def __init__(self, task_config: Dict):
        roi_size = task_config["inferer"]["roi_size"]
        sw_batch_size = task_config["inferer"]["sw_batch_size"]

        self.num_classes = len(task_config["classes"])
        self.classes = task_config["classes"]
        self.fg_classes = get_fg_classes(task_config["condist_config"]["foreground"], task_config["classes"])

        self.inferer = SlidingWindowInferer(
            roi_size=roi_size, sw_batch_size=sw_batch_size, mode="gaussian", overlap=0.5
        )

        self.post = AsDiscreted(
            keys=["label"], to_onehot=[self.num_classes], dim=1
        )
        self.loss_fn = DiceLoss(include_background=False, reduction='none', softmax=True)
        #self.loss_fn = DiceCELoss(include_background=False, reduction='mean', softmax=True)
        self.losses = []

    def validate_step(self, model: torch.nn.Module, batch: Dict[str, Any]) -> None:
        batch["image"] = batch["image"].to("cuda:0")
        batch["label"] = batch["label"].to("cuda:0")

        # Run inference
        batch["preds"] = self.inferer(batch["image"], model)

        # Post processing
        batch = self.post(batch)

        # calculate loss
        loss = self.loss_fn(batch["preds"], batch["label"])  # loss shape: [N, num_classes -1]
        self.losses.append(loss.detach().cpu())

    def validate_loop(self, model, data_loader) -> Dict[str, Any]:
        # Run inference over whole validation set
        with torch.no_grad():
            with autocast('cuda'):
                for batch in tqdm(data_loader, desc="Validation DataLoader", dynamic_ncols=True):
                    self.validate_step(model, batch)

        # Collect losses
        all_losses = torch.cat(self.losses, dim=0)  # shape: [total_samples, num_classes -1]

        mean_loss_per_class = all_losses.mean(dim=0)  # shape: [num_classes -1]
        mean_loss = mean_loss_per_class.mean().item()

        metrics = {}
        # Since include_background=False, class indices correspond to classes 1 to num_classes-1
        for idx, organ in enumerate(self.classes[1:]):  # skip background class
            metrics["val_loss_" + organ] = mean_loss_per_class[idx].item()
        metrics["val_loss"] = mean_loss

        # Clear losses
        self.losses = []

        return metrics

    def run(self, model: torch.nn.Module, data_loader: DataLoader) -> Dict[str, Any]:
        model.eval()
        return self.validate_loop(model, data_loader)

