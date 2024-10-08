from nvflare.apis.dxo import DataKind, from_shareable, DXO
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.abstract.model import ModelLearnable, ModelLearnableKey, model_learnable_to_dxo
from nvflare.app_common.abstract.shareable_generator import ShareableGenerator
from nvflare.app_common.app_constant import AppConstants


class GAShareableGenerator(ShareableGenerator):
    def learnable_to_shareable(self, model_learnable: ModelLearnable, fl_ctx: FLContext) -> Shareable:
        """Convert ModelLearnable to Shareable.

        Args:
            model_learnable (ModelLearnable): model to be converted
            fl_ctx (FLContext): FL context

        Returns:
            Shareable: a shareable containing a DXO object.
        """
        dxo = model_learnable_to_dxo(model_learnable)
        return dxo.to_shareable()

    def shareable_to_learnable(self, shareable: Shareable, fl_ctx: FLContext) -> ModelLearnable:
        """Convert Shareable to ModelLearnable.

        Supporting TYPE == TYPE_WEIGHT_DIFF or TYPE_WEIGHTS or TYPE_COLLECTION

        Args:
            shareable (Shareable): Shareable that contains a DXO object
            fl_ctx (FLContext): FL context

        Returns:
            A ModelLearnable object

        Raises:
            TypeError: if shareable is not of type shareable
            ValueError: if data_kind is not `DataKind.WEIGHTS` and is not `DataKind.WEIGHT_DIFF` and is not `DataKind.COLLECTION`
        """
        if not isinstance(shareable, Shareable):
            raise TypeError("shareable must be Shareable, but got {}.".format(type(shareable)))

        base_model = fl_ctx.get_prop(AppConstants.GLOBAL_MODEL)
        dxo = from_shareable(shareable)
        if dxo.data_kind == DataKind.WEIGHT_DIFF:
            if not base_model:
                self.system_panic(reason="No global base model needed for processing WEIGHT_DIFF!", fl_ctx=fl_ctx)
                return base_model

            weights = base_model[ModelLearnableKey.WEIGHTS]
            if dxo.data is not None:
                model_diff = dxo.data
                for v_name, v_value in model_diff.items():
                    weights[v_name] = weights[v_name] + v_value

        elif dxo.data_kind == DataKind.WEIGHTS:
            if not base_model:
                base_model = ModelLearnable()
            weights = dxo.data
            if not weights:
                self.log_info(fl_ctx, "No model weights found. Model will not be updated.")
            else:
                base_model[ModelLearnableKey.WEIGHTS] = weights

        elif dxo.data_kind == DataKind.COLLECTION:
            
            collection_data = dxo.data
            local_weights = collection_data.get("weights")
            generalization_gap = collection_data.get("generalization_gap")

            if not base_model:
                base_model = ModelLearnable()

            if local_weights:
                base_model[ModelLearnableKey.WEIGHTS] = local_weights
            else:
                self.log_info(fl_ctx, "No model weights found. Model will not be updated.")

            # You can store generalization_gap in base_model for further aggregation
            base_model[ModelLearnableKey.META] = dxo.get_meta_props()
            base_model["generalization_gap"] = generalization_gap

            weights = dxo.data["weights"]
            generalization_gap = dxo.data["generalization_gap"]
        
        else:
            raise ValueError(
                "data_kind should be either DataKind.WEIGHTS or DataKind.WEIGHT_DIFF or DataKind.COLLECTION, but got {}".format(dxo.data_kind)
            )

        base_model[ModelLearnableKey.META] = dxo.get_meta_props()
        return base_model
