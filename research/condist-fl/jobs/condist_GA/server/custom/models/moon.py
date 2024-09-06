from typing import Tuple, Optional, Sequence, Union

import torch
from monai.networks.nets import DynUNet
from monai.networks.blocks import UnetBasicBlock, UnetResBlock
from torch.nn.functional import interpolate

from .mednextv1 import MedNeXt

class SpatialProjection(torch.nn.Module):
    def __init__(
        self,
        filters: Sequence[int],
        norm_name: Union[Tuple, str],
        act_name: Union[Tuple, str],
        res_block: bool = False
    ):
        super().__init__()

        ConvBlock = UnetResBlock if res_block else UnetBasicBlock

        self.proj_layers = torch.nn.ModuleList()
        for i in range(len(filters) - 1):
            self.proj_layers.append(
                ConvBlock(3, filters[i], filters[i+1], 3, 1, norm_name=norm_name, act_name=act_name)
            )

        self.proj_layers.append(
            torch.nn.Conv3d(filters[-1], filters[-1], 1, 1, bias=False)
        )
        self.proj_layers.append(
            torch.nn.InstanceNorm3d(filters[-1], affine=True)
        )

    def forward(self, x):
        for layer in self.proj_layers:
            x = layer(x)
        return x

class MoonMedNeXt(MedNeXt):
    def __init__(
        self,
        in_channels: int,
        n_channels: int,
        n_classes: int,
        exp_r: int = 4,
        kernel_size: int = 7,
        enc_kernel_size: int = None,
        dec_kernel_size: int = None,
        deep_supervision: bool = False,
        do_res: bool = False,
        do_res_up_down: bool = False,
        checkpoint_style: bool = None,
        block_counts: list = [2,2,2,2,2,2,2,2,2],
        norm_type = 'group',
        dim = '3d',
        grn = False
    ):
        super().__init__(
            in_channels,
            n_channels,
            n_classes,
            exp_r,
            kernel_size,
            enc_kernel_size,
            dec_kernel_size,
            deep_supervision,
            do_res,
            do_res_up_down,
            checkpoint_style,
            block_counts,
            norm_type,
            dim,
            grn
        )

        self.proj_layer = SpatialProjection(
            filters=[512, 256],
            norm_name=("INSTANCE", {"affine": True}),
            act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            res_block=False
        )

    def forward(self, x, proj_features: bool = False):
        if not proj_features:
            return super(MoonMedNeXt, self).forward(x)
        else:
            x, skips = self.encode(x)
            v = self.proj_layer(x)
            if self.training and self.do_ds:
                out = self.decode_with_ds(x, skips)
            else:
                out = self.decode(x, skips)
        return out, v

def create_mednextv1_base(num_input_channels, num_classes, kernel_size=3, ds=False):
    return MoonMedNeXt(
        in_channels = num_input_channels,
        n_channels = 32,
        n_classes = num_classes,
        exp_r=[2,3,4,4,4,4,4,3,2],
        kernel_size=kernel_size,
        deep_supervision=ds,
        do_res=True,
        do_res_up_down = True,
        block_counts = [2,2,2,2,2,2,2,2,2],
        checkpoint_style = 'outside_block'
    )
