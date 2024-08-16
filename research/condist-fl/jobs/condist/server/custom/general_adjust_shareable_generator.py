import time
import torch
from nvflare.apis.dxo import DataKind, from_shareable, MetaKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.model import make_model_learnable
from nvflare.app_common.shareablegenerators.full_model_shareable_generator import FullModelShareableGenerator

class GeneralizationAdjustmentShareableGenerator(FullModelShareableGenerator):
    def __init__(self, source_model="model", device=None):
        super().__init__()
        self.source_model = source_model
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.model = None

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        if isinstance(self.source_model, str):
            self.model = engine.get_component(self.source_model)
        else:
            self.model = self.source_model

        if not isinstance(self.model, torch.nn.Module):
            self.system_panic(f"Expected model to be a torch.nn.Module but got {type(self.model)}", fl_ctx)
            return

        self.model.to(self.device)

    def shareable_to_learnable(self, shareable: Shareable, fl_ctx: FLContext):
        dxo = from_shareable(shareable)
        model_diff = dxo.data
        self.model.load_state_dict(model_diff)
        weights = {k: v.detach().cpu().numpy() for k, v in self.model.state_dict().items()}
        return make_model_learnable(weights, dxo.get_meta_props())
