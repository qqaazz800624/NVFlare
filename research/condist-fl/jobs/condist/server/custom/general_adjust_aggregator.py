from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.apis.dxo import from_shareable, DataKind, DXO
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_constant import ReservedKey
from typing import Dict, Union
import numpy as np
from fed_merge import FedAvg

class GeneralizationAdjustmentAggregator(Aggregator):
    def __init__(self, step_size: float = 0.1, exclude_vars: str = None, fair_metric: str = 'loss'):
        """Initialize the Generalization Adjustment Aggregator.

        Args:
            step_size (float): The learning rate for adjusting the aggregation weights.
            exclude_vars (str, optional): Regular expression to exclude variables from aggregation.
            fair_metric (str, optional): Metric for fairness adjustment, either 'acc' for accuracy or 'loss'. Defaults to 'loss'.
        """
        super().__init__()
        self.step_size = step_size
        self.exclude_vars = exclude_vars
        self.fair_metric = fair_metric
        self.client_weights = {}
        self.generalization_gaps = {}
        self.site_results_before_avg = {}
        self.site_results_after_avg = {}
        self.weight_dict = {}

    def compute_generalization_gap(self, site_name: str) -> float:
        """Compute generalization gap for a given client using before and after results."""
        before_metric = self.site_results_before_avg[site_name][self.fair_metric]
        after_metric = self.site_results_after_avg[site_name][self.fair_metric]
        return after_metric - before_metric

    def accept(self, shareable, fl_ctx: FLContext):
        """Receive client updates and prepare for aggregation."""
        dxo = from_shareable(shareable)
        client_name = shareable.get_peer_prop(ReservedKey.IDENTITY_NAME, "?")
        
        # Store model weights for aggregation
        self.client_weights[client_name] = dxo.data

        # Assume site_results_before_avg and site_results_after_avg are collected externally
        # Here we compute the generalization gap for the client
        self.generalization_gaps[client_name] = self.compute_generalization_gap(client_name)

    def refine_weight_dict_by_GA(self):
        """Adjust weights based on the generalization gaps and fairness metric."""
        if self.fair_metric == 'acc':
            signal = -1.0  # To adjust based on accuracy
        elif self.fair_metric == 'loss':
            signal = 1.0  # To adjust based on loss
        else:
            raise ValueError('fair_metric must be acc or loss')

        # Calculate value_list as the difference in metrics (before - after)
        value_list = []
        for site_name in self.site_results_before_avg.keys():
            value_list.append(self.site_results_after_avg[site_name][self.fair_metric] - 
                              self.site_results_before_avg[site_name][self.fair_metric])
        
        value_list = np.array(value_list)
        
        # Normalize the gap list
        norm_gap_list = value_list / np.max(np.abs(value_list))
        
        # Adjust weights for each site
        step_size_adjusted = self.step_size / 3.0  # Adjust step size
        for i, site_name in enumerate(self.weight_dict.keys()):
            self.weight_dict[site_name] += signal * norm_gap_list[i] * step_size_adjusted

        # Normalize the weights to ensure they sum up to 1
        self.weight_dict = self.weight_clip(self.weight_dict)
        
    def weight_clip(self, weight_dict):
        """Clip the weights to ensure they are between 0 and 1, and normalize."""
        new_total_weight = 0.0
        for key_name in weight_dict.keys():
            weight_dict[key_name] = np.clip(weight_dict[key_name], 0.0, 1.0)
            new_total_weight += weight_dict[key_name]
        
        for key_name in weight_dict.keys():
            weight_dict[key_name] /= new_total_weight
        
        return weight_dict

    def aggregate(self, fl_ctx: FLContext):
        """Perform the Generalization Adjustment and aggregate the model updates."""
        # Adjust weights based on generalization gaps
        self.refine_weight_dict_by_GA()
        
        # Perform the FedAvg aggregation using the adjusted weights
        new_model_state = FedAvg(self.client_weights, self.weight_dict)
        
        # Return the aggregated result as a DXO
        aggregated_dxo = DXO(data_kind=DataKind.WEIGHTS, data=new_model_state)
        return aggregated_dxo.to_shareable()
