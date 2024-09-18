from typing import Any, Dict

import torch
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceLoss, DiceCELoss, DeepSupervisionLoss
from monai.transforms import AsDiscreted
from monai.metrics import DiceMetric
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import autocast
from losses import MarginalDiceCELoss, ConDistDiceLoss


def get_fg_classes(fg_idx, classes):
    out = {}
    for idx in fg_idx:
        out[classes[idx]] = idx
    return out


class Validator_loss(object):
    def __init__(self, task_config: Dict):
        roi_size = task_config["inferer"]["roi_size"]
        sw_batch_size = task_config["inferer"]["sw_batch_size"]

        self.num_classes = len(task_config["classes"])
        self.classes = task_config["classes"]
        foreground = task_config["condist_config"]["foreground"]
        self.fg_classes = get_fg_classes(task_config["condist_config"]["foreground"], task_config["classes"])

        self.inferer = SlidingWindowInferer(
            roi_size=roi_size, sw_batch_size=sw_batch_size, mode="gaussian", overlap=0.5
        )

        self.post = AsDiscreted(
            keys=["label"], to_onehot=[self.num_classes], dim=1
        )
        # self.post = AsDiscreted(
        #     keys=["preds", "label"], argmax=[True, False], to_onehot=[self.num_classes, self.num_classes], dim=1
        # )
        
        #self.loss_fn = DiceLoss(include_background=False, reduction='none', softmax=True)
        self.marginal_loss_fn = MarginalDiceCELoss(foreground, softmax=True, smooth_nr=0.0, batch=True)
        self.ds_loss_fn = DeepSupervisionLoss(self.marginal_loss_fn, weights=[0.5333, 0.2667, 0.1333, 0.0667])
        self.loss_fn = DiceCELoss(include_background=False, reduction='mean', softmax=True)
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
        #loss = self.marginal_loss_fn(batch["preds"], batch["label"])  # loss shape: [N, num_classes -1]
        #print('shape of validation batch["preds"]: ', batch["preds"].shape)
        # print('dim of batch["preds"]: ', batch["preds"].dim())
        #preds = [batch["preds"][:, i, ::] for i in range(batch["preds"].shape[1])]
        #print('len of validation preds: ', len(preds))
        #print('shape of validation preds: ', preds[0].shape)
        #print('shape of validation batch["label"]: ', batch["label"].shape)
        #loss = self.ds_loss_fn(preds, batch["label"])
        self.losses.append(loss.detach().cpu())


    def validate_loop(self, model, data_loader) -> Dict[str, Any]:
        # Run inference over whole validation set
        with torch.no_grad():
            with autocast('cuda'):
                for batch in tqdm(data_loader, desc="Validation DataLoader", dynamic_ncols=True):
                    self.validate_step(model, batch)

        # Collect losses
        # all_losses = torch.cat(self.losses, dim=0)  # shape: [total_samples, num_classes -1]
        # mean_loss_per_class = all_losses.mean(dim=0)  # shape: [num_classes -1]
        # mean_loss = mean_loss_per_class.mean().item()
        all_losses = torch.stack(self.losses)
        mean_loss = all_losses.mean().item()
        
        # Collect metrics
        
        metrics = {}
        metrics["val_loss"] = mean_loss

        # Clear losses
        self.losses = []

        return metrics

    def run(self, model: torch.nn.Module, data_loader: DataLoader) -> Dict[str, Any]:
        model.eval()
        return self.validate_loop(model, data_loader)

