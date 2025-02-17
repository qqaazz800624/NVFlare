from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.apis.dxo import from_shareable, DataKind, DXO
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.fl_constant import ReservedKey
from typing import Dict, Union
import numpy as np
from fed_merge import FedAvg
from nvflare.app_common.app_constant import AppConstants

class GeneralizationAdjustmentAggregator(Aggregator):
    def __init__(self, num_clients: int = 4, step_size: float = 0.5, exclude_vars: str = None, fair_metric: str = 'loss'):
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
        self.aggregation_weights = {} # This will be used to store the weights for aggregation
        self.num_clients = num_clients

    def accept(self, shareable, fl_ctx: FLContext):
        """Receive client updates and prepare for aggregation."""
        dxo = from_shareable(shareable)
        client_name = shareable.get_peer_prop(ReservedKey.IDENTITY_NAME, "?")

        # Check if the DXO contains a collection of weights and generalization gap
        if dxo.data_kind == DataKind.COLLECTION:
            # Extract the weights and generalization gap
            self.client_weights[client_name] = dxo.data.get("weights", {})
            self.generalization_gaps[client_name] = dxo.data.get("generalization_gap", 0.0)

            # Initialize aggregation_weights with equal weights if it's the first round
            if not self.aggregation_weights:
                self.aggregation_weights = {client: 1.0 / self.num_clients for client in self.client_weights}
                self.log_info(fl_ctx, f"Initialized aggregation_weights with equal weights: {self.aggregation_weights}")
        else:
            # If it's not a collection, log an error
            self.log_error(fl_ctx, f"Expected DataKind.COLLECTION, but received {dxo.data_kind}.")
        
        # Optionally log the acceptance of the model weights and generalization gaps
        self.log_info(fl_ctx, f"Accepted weights and generalization gap from client: {client_name}")


    def refine_weight_dict_by_GA(self, fl_ctx: FLContext):
        """Adjust weights based on the generalization gaps and fairness metric."""
        if self.fair_metric == 'acc':
            signal = -1.0  # To adjust based on accuracy
        elif self.fair_metric == 'loss':
            signal = 1.0  # To adjust based on loss
        else:
            raise ValueError('fair_metric must be acc or loss')
        
        # Retrieve the current round and total number of rounds
        current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND, default=1)
        num_rounds = fl_ctx.get_prop(AppConstants.NUM_ROUNDS, default=120)
        self.log_info(fl_ctx, f"Current Round: {current_round}, Total Rounds: {num_rounds}")

        # Calculate value_list as the generalization gap received from clients
        value_list = np.array([self.generalization_gaps[site_name] for site_name in self.generalization_gaps.keys()])
        self.log_info(fl_ctx, f"Generalization Gaps: {value_list}")

        # Compute the mean of the generalization gaps
        mean_generalization_gap = np.mean(value_list)
        self.log_info(fl_ctx, f"Mean Generalization Gap: {mean_generalization_gap}")

        centralized_value_list = value_list - mean_generalization_gap

        # Avoid division by zero if all generalization gaps are the same
        max_abs_value = np.max(np.abs(centralized_value_list))

        if max_abs_value == 0:
            norm_gap_list = np.zeros_like(centralized_value_list)
        else:
            # Normalize the gap list
            norm_gap_list = centralized_value_list / max_abs_value

        # Adjust weights for each site
        step_size_adjusted = (1-(current_round/num_rounds))*self.step_size   # Adjust step size

        for i, site_name in enumerate(self.aggregation_weights.keys()):
            self.weight_aggregation_weightsdict[site_name] += signal * norm_gap_list[i] * step_size_adjusted

        # Normalize the weights to ensure they sum up to 1
        self.aggregation_weights = self.weight_clip(self.aggregation_weights)

        
    def weight_clip(self, aggregation_weights):
        """Clip the weights to ensure they are between 0 and 1, and normalize."""
        new_total_weight = 0.0
        for key_name in aggregation_weights.keys():
            aggregation_weights[key_name] = np.clip(aggregation_weights[key_name], 0.0, 1.0)
            new_total_weight += aggregation_weights[key_name]
        
        if new_total_weight > 0:
            for key_name in aggregation_weights.keys():
                aggregation_weights[key_name] /= new_total_weight
        
        return aggregation_weights

    def aggregate(self, fl_ctx: FLContext):
        """Perform the Generalization Adjustment and aggregate the model updates."""
        
        # Adjust weights based on generalization gaps
        self.refine_weight_dict_by_GA(fl_ctx)
        
        # Log the weights being used for aggregation
        self.log_info(fl_ctx, f"Aggregating using weights: {self.aggregation_weights}")
        
        # Perform the FedAvg aggregation using the adjusted weights
        new_model_state = FedAvg(self.client_weights, self.aggregation_weights)
        
        # Return the aggregated result as a DXO
        aggregated_dxo = DXO(data_kind=DataKind.WEIGHTS, data=new_model_state)
        
        # Optionally log the successful aggregation
        self.log_info(fl_ctx, "Model aggregation complete. Returning aggregated model.")
        
        return aggregated_dxo.to_shareable()


