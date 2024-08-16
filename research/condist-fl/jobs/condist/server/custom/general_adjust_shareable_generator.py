import time
import torch
from nvflare.apis.dxo import DataKind, MetaKey, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.learnable import Learnable
from nvflare.app_common.abstract.model import ModelLearnableKey, make_model_learnable
from nvflare.app_common.shareablegenerators.full_model_shareable_generator import FullModelShareableGenerator

class GeneralizationAdjustmentShareableGenerator(FullModelShareableGenerator):
    def __init__(self, optimizer_args: dict = None, lr_scheduler_args: dict = None, step_size: float = 0.1, device=None):
        """Custom Shareable Generator implementing Generalization Adjustment."""
        super().__init__()
        self.optimizer_args = optimizer_args
        self.lr_scheduler_args = lr_scheduler_args
        self.step_size = step_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            # Initialize the model and optimizer
            engine = fl_ctx.get_engine()
            self.model = engine.get_component(self.source_model)
            self.model.to(self.device)
            self._initialize_optimizer(fl_ctx)

    def _initialize_optimizer(self, fl_ctx: FLContext):
        # Initialize optimizer
        engine = fl_ctx.get_engine()
        self.optimizer_args["args"]["params"] = self.model.parameters()
        self.optimizer = engine.build_component(self.optimizer_args)
        if self.lr_scheduler_args:
            self.lr_scheduler_args["args"]["optimizer"] = self.optimizer
            self.lr_scheduler = engine.build_component(self.lr_scheduler_args)

    def server_update(self, model_diff):
        """Update the global model using the specified optimizer."""
        self.model.train()
        self.optimizer.zero_grad()
        for name, param in self.model.named_parameters():
            if name in model_diff:
                param.grad = torch.tensor(-1.0 * model_diff[name]).to(self.device)
        self.optimizer.step()
        if self.lr_scheduler:
            self.lr_scheduler.step()

        return self.model.state_dict()

    def shareable_to_learnable(self, shareable: Shareable, fl_ctx: FLContext) -> Learnable:
        """Convert Shareable to Learnable, applying Generalization Adjustment."""
        dxo = from_shareable(shareable)
        if dxo.data_kind != DataKind.WEIGHT_DIFF:
            self.system_panic("Invalid data kind for this operation", fl_ctx)
            return Learnable()

        model_diff = dxo.data
        weights = self.server_update(model_diff)
        for key in weights:
            weights[key] = weights[key].detach().cpu().numpy()

        return make_model_learnable(weights, dxo.get_meta_props())
