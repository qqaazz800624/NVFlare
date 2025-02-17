from typing import Dict
from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.aggregators.assembler import Assembler
import numpy as np
from nvflare.app_common.app_constant import AppConstants

class GeneralizationAdjustmentAssembler(Assembler):
    def __init__(self, num_clients: int = 4, step_size: float = 0.5, fair_metric: str = 'loss'):
        super().__init__(data_kind=DataKind.COLLECTION)
        self.step_size = step_size
        self.fair_metric = fair_metric
        self.num_clients = num_clients
        self.aggregation_weights = None

    def get_model_params(self, dxo: DXO) -> dict:
        """Extracts model weights and generalization gap from the DXO."""
        return {
            "weights": dxo.data.get("weights", {}),
            "generalization_gap": dxo.data.get("generalization_gap", 0.0)
        }

    def assemble(self, data: Dict[str, dict], fl_ctx: FLContext) -> DXO:
        """Assemble the collected submissions based on Generalization Adjustment."""
        current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND, default=0)

        if current_round == 0:
            # Initialize equal weights for round 0
            self.aggregation_weights = {client: 1.0 / self.num_clients for client in data.keys()}
            self.log_info(fl_ctx, "Initialized equal aggregation weights for round 0")
        else:
            # Perform Generalization Adjustment starting from round 1
            generalization_gaps = np.array([info["generalization_gap"] for info in data.values()])
            generalization_gaps_dict = {site_name: info["generalization_gap"] for site_name, info in data.items()}
            self.log_info(fl_ctx, f"Generalization Gaps: {generalization_gaps_dict}")
            #self.log_info(fl_ctx, f"Generalization Gaps: {generalization_gaps}")
            
            mean_generalization_gap = np.mean(generalization_gaps)
            centralized_gaps = generalization_gaps - mean_generalization_gap

            max_abs_value = np.max(np.abs(centralized_gaps))
            norm_gap_list = centralized_gaps / max_abs_value if max_abs_value != 0 else np.zeros_like(centralized_gaps)

            step_size_adjusted = (1 - (current_round / fl_ctx.get_prop(AppConstants.NUM_ROUNDS, default=120))) * self.step_size

            for i, site_name in enumerate(data.keys()):
                signal = -1.0 if self.fair_metric == 'acc' else 1.0
                self.aggregation_weights[site_name] += signal * norm_gap_list[i] * step_size_adjusted

            # Normalize the weights to ensure they sum up to 1
            total_weight = sum(self.aggregation_weights.values())
            if total_weight > 0:
                for key in self.aggregation_weights.keys():
                    self.aggregation_weights[key] = np.clip(self.aggregation_weights[key], 0.0, 1.0)
                    self.aggregation_weights[key] /= total_weight

            self.log_info(fl_ctx, f"Adjusted aggregation weights based on Generalization Adjustment: {self.aggregation_weights}")

        # Perform the weighted aggregation
        aggregated_weights = {}
        for var_name in data[next(iter(data))]["weights"].keys():
            aggregated_weights[var_name] = sum(
                info["weights"][var_name] * self.aggregation_weights[site_name] for site_name, info in data.items()
            )

        # Create and return a DXO containing the aggregated weights
        return DXO(data_kind=DataKind.WEIGHTS, data=aggregated_weights)
