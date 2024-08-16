import numpy as np
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.fl_context import FLContext
import torch

class GeneralizationAdjustmentAggregator(Aggregator):
    def __init__(self, step_size=0.1, exclude_vars=None):
        super().__init__()
        self.step_size = step_size
        self.exclude_vars = exclude_vars
        self.client_weights = {}
        self.generalization_gaps = {}
        self.site_results_before_avg = {}
        self.site_results_after_avg = {}

    def refine_weight_dict_by_GA(self, weight_dict, site_before_results_dict, site_after_results_dict, step_size=0.1, fair_metric='loss'):
        if fair_metric == 'acc':
            signal = -1.0
        elif fair_metric == 'loss':
            signal = 1.0
        else:
            raise ValueError('fair_metric must be acc or loss')
    
        value_list = []
        for site_name in site_before_results_dict.keys():
            value_list.append(site_after_results_dict[site_name][fair_metric] - site_before_results_dict[site_name][fair_metric])
        
        value_list = np.array(value_list)
        
        
        step_size = 1./3. * step_size
        norm_gap_list = value_list / np.max(np.abs(value_list))
        
        for i, site_name in enumerate(weight_dict.keys()):
            weight_dict[site_name] += signal * norm_gap_list[i] * step_size

        weight_dict = self.weight_clip(weight_dict)
        
        return weight_dict
    
    def weight_clip(self, weight_dict):
        new_total_weight = 0.0
        for key_name in weight_dict.keys():
            weight_dict[key_name] = np.clip(weight_dict[key_name], 0.0, 1.0)
            new_total_weight += weight_dict[key_name]
        
        for key_name in weight_dict.keys():
            weight_dict[key_name] /= new_total_weight
        
        return weight_dict
    
    def site_evaluation(self, epochs, site_name, args, model, dataloader, log_file, log_ten, metric, note='after_fed'):
        model.eval()
        with torch.no_grad():
            for imgs, labels, domain_labels, in dataloader:
                imgs = imgs.cuda()
                output = model(imgs)
                metric.update(output, labels)
        results_dict = metric.results()
        log_ten.add_scalar(f'{note}_{site_name}_loss', results_dict['loss'], epochs)
        log_ten.add_scalar(f'{note}_{site_name}_acc', results_dict['acc'], epochs)
        log_file.info(f'{note} Round: {epochs:3d} | Epochs: {args.local_epochs*epochs:3d} | Domain: {site_name} | loss: {results_dict["loss"]:.4f} | Acc: {results_dict["acc"]*100:.2f}%')

        return results_dict

    def compute_generalization_gap(self, site_results_before, site_results_after, metric='loss'):
        """Compute the generalization gap for a given site."""
        return site_results_after[metric] - site_results_before[metric]

    def accept(self, shareable, fl_ctx: FLContext):
        dxo = from_shareable(shareable)
        client_name = shareable.get_peer_prop("client_name", "?")
        round_num = shareable.get_cookie("round_number")

        self.client_weights[client_name] = dxo.data  # Store client model weights
        return True

    def aggregate(self, fl_ctx: FLContext):
        # Perform evaluations on client models before aggregation
        for client_name in self.client_weights.keys():
            self.site_results_before_avg[client_name] = self.site_evaluation(client_name)  # Before aggregation

        # Perform FedAvg aggregation
        aggregated_weights = self.perform_fed_avg()

        # Perform evaluations on aggregated model after aggregation
        for client_name in self.client_weights.keys():
            self.site_results_after_avg[client_name] = self.site_evaluation(client_name)  # After aggregation

        # Adjust the client weights using the Generalization Adjustment method
        adjusted_weights = self.refine_weight_dict_by_GA(self.client_weights, 
                                                    self.site_results_before_avg, 
                                                    self.site_results_after_avg, 
                                                    step_size=self.step_size)

        # Aggregate the models with adjusted weights
        final_aggregated_weights = self.perform_final_aggregation(adjusted_weights)

        # Return the final aggregated result as a DXO
        aggregated_dxo = DXO(data_kind=DataKind.WEIGHTS, data=final_aggregated_weights)
        return aggregated_dxo.to_shareable()

    def perform_fed_avg(self):
        """Perform FedAvg aggregation."""
        aggregated_weights = {}
        total_weight = sum(self.client_weights.values())
        for client_name, client_weight in self.client_weights.items():
            for k, v in client_weight.items():
                if k not in aggregated_weights:
                    aggregated_weights[k] = v * (self.client_weights[client_name] / total_weight)
                else:
                    aggregated_weights[k] += v * (self.client_weights[client_name] / total_weight)
        return aggregated_weights

    def perform_final_aggregation(self, adjusted_weights):
        """Perform final aggregation using adjusted weights."""
        aggregated_weights = {}
        total_weight = sum(adjusted_weights.values())
        for client_name, client_weight in adjusted_weights.items():
            for k, v in self.client_weights[client_name].items():
                if k not in aggregated_weights:
                    aggregated_weights[k] = v * (adjusted_weights[client_name] / total_weight)
                else:
                    aggregated_weights[k] += v * (adjusted_weights[client_name] / total_weight)
        return aggregated_weights
