# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict

import torch
from monai.inferers import SlidingWindowInferer
from monai.metrics import DiceMetric
from monai.transforms import AsDiscreted
from torch.utils.data import DataLoader
from tqdm import tqdm
#from losses import ConDistDiceLoss
from losses import MarginalDiceCELoss
from monai.losses import DeepSupervisionLoss

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
        self.fg_classes = get_fg_classes(task_config["condist_config"]["foreground"], task_config["classes"])

        self.inferer = SlidingWindowInferer(
            roi_size=roi_size, sw_batch_size=sw_batch_size, mode="gaussian", overlap=0.5
        )

        self.post = AsDiscreted(
            keys=["preds", "label"], argmax=[True, False], to_onehot=[self.num_classes, self.num_classes], dim=1
        )
        self.metric = DiceMetric(reduction="mean_batch")

        # Initialize loss functions as in ConDistTrainer
        # self.condist_loss_fn = ConDistDiceLoss(
        #     num_classes=self.num_classes,
        #     foreground=task_config["condist_config"]["foreground"],
        #     background=task_config["condist_config"]["background"],
        #     temperature=task_config["condist_config"].get("temperature", 2.0),
        #     smooth_nr=0.0,
        #     batch=True
        # )

        # Initialize only the Deep Supervision Loss function
        self.marginal_loss_fn = MarginalDiceCELoss(foreground=task_config["condist_config"]["foreground"], softmax=True, smooth_nr=0.0, batch=True)
        self.ds_loss_fn = DeepSupervisionLoss(self.marginal_loss_fn, weights=[0.5333, 0.2667, 0.1333, 0.0667])

    def validate_step(self, model: torch.nn.Module, batch: Dict[str, Any]) -> None:
        batch["image"] = batch["image"].to("cuda:0")
        batch["label"] = batch["label"].to("cuda:0")

        # Run inference
        batch["preds"] = self.inferer(batch["image"], model)

        # Post processing
        batch = self.post(batch)

        # Calculate Deep Supervision Loss
        if batch["preds"].dim() == 6:
            preds_list = [batch["preds"][:, i, ::] for i in range(batch["preds"].shape[1])]
        else:
            preds_list = [batch["preds"]]

        ds_loss = self.ds_loss_fn(preds_list, batch["label"])

        # # Calculate ConDist Loss
        # with torch.no_grad():
        #     targets = global_model(batch["image"])
        #     if targets.dim() == 6:
        #         targets = targets[:, 0, ::]
        # condist_loss = self.condist_loss_fn(preds_list[0], targets, batch["label"])

        # calculate metrics
        self.metric(batch["preds"], batch["label"])

        return ds_loss.item()

    def validate_loop(self, model: torch.nn.Module, data_loader: DataLoader, global_model: torch.nn.Module) -> Dict[str, Any]:
        total_ds_loss = 0.0
        #total_condist_loss = 0.0
        count = 0

        # Run inference over whole validation set
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                for batch in tqdm(data_loader, desc="Validation DataLoader", dynamic_ncols=True):
                    ds_loss = self.validate_step(model, batch)
                    total_ds_loss += ds_loss
                    count += 1

        # Calculate average losses
        avg_ds_loss = total_ds_loss / count
        #avg_condist_loss = total_condist_loss / count

        # Collect metrics
        raw_metrics = self.metric.aggregate()
        self.metric.reset()

        mean = 0.0
        metrics = {}
        for organ, idx in self.fg_classes.items():
            mean += raw_metrics[idx]
            metrics["val_meandice_" + organ] = raw_metrics[idx]
        metrics["val_meandice"] = mean / len(self.fg_classes)

        # Add the average Deep Supervision Loss to the metrics
        metrics["avg_ds_loss"] = avg_ds_loss
        #metrics["avg_condist_loss"] = avg_condist_loss

        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                metrics[k] = v.tolist()
        return metrics

    def run(self, model: torch.nn.Module, data_loader: DataLoader) -> Dict[str, Any]:
        model.eval()
        #global_model.eval()
        return self.validate_loop(model, data_loader)
