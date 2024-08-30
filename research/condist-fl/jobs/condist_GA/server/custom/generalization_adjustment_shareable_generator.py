import time
import torch
from nvflare.apis.dxo import DataKind, MetaKey, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.learnable import Learnable
from nvflare.app_common.abstract.model import ModelLearnableKey, make_model_learnable
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.shareablegenerators.full_model_shareable_generator import FullModelShareableGenerator
from nvflare.security.logging import secure_format_exception

class GeneralizationAdjustmentShareableGenerator(FullModelShareableGenerator):
    def __init__(self, source_model="model", device=None):
        """Implement Generalization Adjustment with DataKind.COLLECTION."""
        super().__init__()
        self.source_model = source_model
        self.model = None
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            # Initialize the model
            engine = fl_ctx.get_engine()
            if isinstance(self.source_model, str):
                self.model = engine.get_component(self.source_model)
            else:
                self.model = self.source_model

            if self.model is None:
                self.system_panic("Model is not available", fl_ctx)
                return
            elif not isinstance(self.model, torch.nn.Module):
                self.system_panic(f"Expected model to be a torch.nn.Module but got {type(self.model)}", fl_ctx)
                return

            self.model.to(self.device)

    def server_update(self, weights, generalization_gaps):
        """Update the global model using the Generalization Adjustment strategy.

        Args:
            weights: The aggregated weights from clients.
            generalization_gaps: The generalization gaps from clients.

        Returns:
            Updated model state dictionary.
        """
        self.model.train()
        # Implement the logic for adjusting the model based on the weights and generalization gaps.
        # This is where the specifics of your Generalization Adjustment algorithm will be implemented.

        # For example:
        updated_params = []
        for name, param in self.model.named_parameters():
            if name in weights:
                param.data = torch.tensor(weights[name]).to(self.device)
                updated_params.append(name)
        
        # Incorporate generalization gaps if needed here.
        # This might involve modifying the weights or adjusting the parameters in some way.
        
        return self.model.state_dict(), updated_params

    def shareable_to_learnable(self, shareable: Shareable, fl_ctx: FLContext) -> Learnable:
        """Convert Shareable to Learnable while doing a Generalization Adjustment update.

        Supporting data_kind == DataKind.COLLECTION.

        Args:
            shareable (Shareable): Shareable to be converted
            fl_ctx (FLContext): FL context

        Returns:
            Model: Updated global ModelLearnable.
        """
        # Extract DXO from Shareable
        dxo = from_shareable(shareable)

        if dxo.data_kind != DataKind.COLLECTION:
            self.system_panic("Expected data_kind to be DataKind.COLLECTION", fl_ctx)
            return Learnable()

        data = dxo.data
        weights = data.get("weights")
        generalization_gaps = data.get("generalization_gap")

        start = time.time()
        updated_weights, updated_params = self.server_update(weights, generalization_gaps)
        secs = time.time() - start

        # Convert to numpy dict of weights
        start = time.time()
        for key in updated_weights:
            updated_weights[key] = updated_weights[key].detach().cpu().numpy()
        secs_detach = time.time() - start

        # Log the update process
        self.log_info(fl_ctx, f"Generalization Adjustment server model update round {fl_ctx.get_prop(AppConstants.CURRENT_ROUND)}, update: {secs} secs., detach: {secs_detach} secs.")

        return make_model_learnable(updated_weights, dxo.get_meta_props())
