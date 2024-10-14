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
        self.fg_classes = get_fg_classes(task_config["condist_config"]["foreground"], task_config["classes"])

        self.inferer = SlidingWindowInferer(
            roi_size=roi_size, sw_batch_size=sw_batch_size, mode="gaussian", overlap=0.5
        )

        self.post = AsDiscreted(
            keys=["preds", "label"], argmax=[True, False], to_onehot=[self.num_classes, self.num_classes], dim=1
        )
        #self.metric = DiceMetric(reduction="mean_batch")
        self.metric = DiceMetric(include_background=False, reduction="none")

    def validate_step(self, model: torch.nn.Module, batch: Dict[str, Any]) -> None:
        batch["image"] = batch["image"].to("cuda:0")
        batch["label"] = batch["label"].to("cuda:0")

        # Run inference
        batch["preds"] = self.inferer(batch["image"], model)

        # Post processing
        batch = self.post(batch)

        # calculate metrics
        self.metric(batch["preds"], batch["label"])

    def validate_loop(self, model, data_loader) -> Dict[str, Any]:
        
        # List to store per-sample dice scores
        per_sample_dice = []
        
        # Run inference over whole validation set
        with torch.no_grad():
            #with torch.cuda.amp.autocast():
            with autocast('cuda'):
                for batch in tqdm(data_loader, desc="Validation DataLoader", dynamic_ncols=True):
                    self.validate_step(model, batch)

        # Collect metrics
        raw_metrics = self.metric.aggregate()
        self.metric.reset()

        # Convert raw_metrics to a tensor if it's not already
        if not isinstance(raw_metrics, torch.Tensor):
            raw_metrics = torch.tensor(raw_metrics)

        metrics = {}
        mean_per_class = {}
        var_per_class = {}

        for organ, idx in self.fg_classes.items():
            organ_scores = raw_metrics[:, idx]
            organ_mean = torch.mean(organ_scores)
            organ_variance = torch.var(organ_scores)

            metrics[f"val_meandice_{organ}"] = organ_mean.item()
            metrics[f"val_vardice_{organ}"] = organ_variance.item()

            mean_per_class[organ] = organ_mean
            var_per_class[organ] = organ_variance

        # Overall mean and variance across classes
        mean_values = torch.stack(list(mean_per_class.values()))
        var_values = torch.stack(list(var_per_class.values()))

        metrics["val_meandice"] = torch.mean(mean_values).item()
        metrics["val_vardice"] = torch.mean(var_values).item()

        # Optional: Compute overall variance across all scores
        all_scores = raw_metrics[:, list(self.fg_classes.values())].flatten()
        overall_mean = torch.mean(all_scores)
        overall_var = torch.var(all_scores, unbiased=True)

        metrics["val_meandice_all"] = overall_mean.item()
        metrics["val_vardice_all"] = overall_var.item()
        
        return metrics

    def run(self, model: torch.nn.Module, data_loader: DataLoader) -> Dict[str, Any]:
        model.eval()
        return self.validate_loop(model, data_loader)
