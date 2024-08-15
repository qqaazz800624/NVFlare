from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.apis.dxo import from_shareable, DataKind, DXO
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_constant import ReservedKey
from typing import Dict, Union

class GeneralizationAdjustmentAggregator(Aggregator):
    def __init__(self, step_size: float, exclude_vars: str = None):
        """Initialize the Generalization Adjustment Aggregator.

        Args:
            step_size (float): The learning rate for adjusting the aggregation weights.
            exclude_vars (str, optional): Regular expression to exclude variables from aggregation.
        """
        super().__init__()
        self.step_size = step_size
        self.exclude_vars = exclude_vars
        self.client_weights = {}
        self.generalization_gaps = {}

    def compute_generalization_gap(self, client_data: Dict) -> float:
        """Compute generalization gap for a given client."""
        # Implement your method to compute the generalization gap for the client
        return client_data.get('generalization_gap', 0.0)

    def accept(self, shareable, fl_ctx: FLContext):
        """Receive client updates and prepare for aggregation."""
        dxo = from_shareable(shareable)
        client_name = shareable.get_peer_prop(ReservedKey.IDENTITY_NAME, "?")
        round_num = shareable.get_cookie("round_number")
        
        # Extract generalization gap
        generalization_gap = self.compute_generalization_gap(dxo.data)
        self.generalization_gaps[client_name] = generalization_gap

        # Store model weights (dxo.data) for aggregation
        self.client_weights[client_name] = dxo.data

    def aggregate(self, fl_ctx: FLContext):
        """Perform the Generalization Adjustment and aggregate the model updates."""
        # Compute the mean generalization gap (mu)
        mean_gap = sum(self.generalization_gaps.values()) / len(self.generalization_gaps)
        
        # Compute the maximum generalization gap (max_gap)
        max_gap = max(self.generalization_gaps.values())
        
        # Initialize total weight for normalization
        total_weight = 0
        adjusted_weights = {}

        # Adjust weights using generalization gaps
        for client, gap in self.generalization_gaps.items():
            # (G_Di(theta_r) - mu) * d_r / max_gap + a_i
            adjustment_factor = ((gap - mean_gap) * self.step_size) / max_gap
            adjusted_weight = self.client_weights[client] + adjustment_factor
            
            # Store adjusted weight
            adjusted_weights[client] = adjusted_weight
            total_weight += adjusted_weight

        # Normalize weights and perform weighted aggregation
        aggregated_result = {}
        for client, weight in adjusted_weights.items():
            normalized_weight = weight / total_weight
            for key, value in self.client_weights[client].items():
                if key in aggregated_result:
                    aggregated_result[key] += value * normalized_weight
                else:
                    aggregated_result[key] = value * normalized_weight

        # Return the aggregated result as a DXO
        aggregated_dxo = DXO(data_kind=DataKind.WEIGHTS, data=aggregated_result)
        return aggregated_dxo.to_shareable()


