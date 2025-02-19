from typing import Any, Dict

import torch
from monai.inferers import SlidingWindowInferer
from monai.losses import DiceLoss, DiceCELoss, DeepSupervisionLoss
from monai.transforms import AsDiscreted
from monai.metrics import DiceMetric
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import autocast
from losses import MarginalDiceCELoss, ConDistDiceLoss, MarginalDiceLoss
from loss_evidential import MarginalEvidentialLoss, MaskedEvidentialLoss
from utils.get_model import get_model

def get_fg_classes(fg_idx, classes):
    out = {}
    for idx in fg_idx:
        out[classes[idx]] = idx
    return out


class Validator_loss_local(object):
    def __init__(self, task_config: Dict):
        roi_size = task_config["inferer"]["roi_size"]
        sw_batch_size = task_config["inferer"]["sw_batch_size"]

        self.num_classes = len(task_config["classes"])
        self.classes = task_config["classes"]
        foreground = task_config["condist_config"]["foreground"]
        background = task_config["condist_config"]["background"]
        temperature = task_config["condist_config"].get("temperature", 2.0)
        self.max_rounds = task_config["training"]["max_rounds"]
        self.current_round = 0
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

        self.weight_range = task_config["condist_config"]["weight_schedule_range"]
        
        self.marginal_loss_fn = MarginalDiceCELoss(foreground, softmax=True, smooth_nr=0.0, batch=True)
        self.condist_loss_fn = ConDistDiceLoss(
            self.num_classes, foreground, background, temperature=temperature, smooth_nr=0.0, batch=True
        )
        self.losses = []
    
    def update_condist_weight(self, current_round):
        left = min(self.weight_range)
        right = max(self.weight_range)
        intv = (right - left) / (self.max_rounds - 1)
        self.weight = left + intv * current_round

    def validate_step(self, model: torch.nn.Module, batch: Dict[str, Any], global_model: torch.nn.Module, current_round) -> None:
        batch["image"] = batch["image"].to("cuda:0")
        batch["label"] = batch["label"].to("cuda:0")

        # Run inference
        batch["preds"] = self.inferer(batch["image"], model)

        # Run global model's inference
        batch["targets"] = self.inferer(batch["image"], global_model)

        # Post processing
        # batch = self.post(batch)

        # calculate loss
        #loss = self.loss_fn(batch["preds"], batch["label"])  # loss shape: [N, num_classes -1]
        marginal_loss = self.marginal_loss_fn(batch["preds"], batch["label"])  
        condist_loss = self.condist_loss_fn(batch["preds"], batch["targets"], batch["label"])
        #marginal_evidential_loss = self.evidential_loss_fn(batch["preds"], batch["label"], current_round)
        #masked_evidential_loss = self.masked_evidential_loss_fn(batch["preds"], batch["label"])
        self.update_condist_weight(current_round)
        loss = marginal_loss - self.weight * condist_loss
        self.losses.append(loss.detach().cpu())


    def validate_loop(self, model, data_loader, global_model, current_round) -> Dict[str, Any]:
        # Run inference over whole validation set
        with torch.no_grad():
            with autocast('cuda'):
                for batch in tqdm(data_loader, desc="Validation DataLoader", dynamic_ncols=True):
                    self.validate_step(model, batch, global_model, current_round)

        # Collect losses
        all_losses = torch.stack(self.losses)
        mean_loss = all_losses.mean().item()
        
        # Collect metrics
        
        metrics = {}
        metrics["val_loss"] = mean_loss

        # Clear losses
        self.losses = []

        return metrics

    def run(self, model: torch.nn.Module, data_loader: DataLoader, global_model: torch.nn.Module, current_round) -> Dict[str, Any]:
        model.eval()
        global_model.eval()
        return self.validate_loop(model, data_loader, global_model, current_round)

