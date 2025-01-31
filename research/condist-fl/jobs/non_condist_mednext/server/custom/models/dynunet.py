from typing import Tuple, Optional, Sequence, Union

import torch
from monai.networks.nets import DynUNet
from monai.networks.blocks import UnetBasicBlock, UnetResBlock
from torch.nn.functional import interpolate


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

class MoonDynUNet(DynUNet):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[Union[Sequence[int], int]],
        strides: Sequence[Union[Sequence[int], int]],
        upsample_kernel_size: Sequence[Union[Sequence[int], int]],
        filters: Optional[Sequence[int]] = None,
        dropout: Optional[Union[Tuple, str, float]] = None,
        norm_name: Union[Tuple, str] = ("INSTANCE", {"affine": True}),
        act_name: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        deep_supervision: bool = False,
        deep_supr_num: int = 1,
        res_block: bool = False,
        trans_bias: bool = False
    ):
        super(MoonDynUNet, self).__init__(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size,
            strides,
            upsample_kernel_size,
            filters=filters,
            dropout=dropout,
            norm_name=norm_name,
            act_name=act_name,
            deep_supervision=deep_supervision,
            deep_supr_num=deep_supr_num,
            res_block=res_block,
            trans_bias=trans_bias
        )

        f = self.filters[-1]
        if 256 <= f <= 320:
            proj_filters = [f, 256]
        else:
            proj_filters = [f, 128]

        self.proj_layer = SpatialProjection(
            filters=proj_filters,
            norm_name=norm_name,
            act_name=act_name,
            res_block=res_block
        )

    def encode(self, x):
        out = self.input_block(x)
        skips = [out]
        for block in self.downsamples:
            out = block(out)
            skips.append(out)
        out = self.bottleneck(out)
        return out, skips

    def decode(self, x, skips):
        temp = []
        skips = skips[::-1]
        for i, block in enumerate(self.upsamples):
            x = block(x, skips[i])
            temp.append(x)
        out = self.output_block(x)

        # Construct deep supervision output
        if self.training and self.deep_supervision:
            temp = temp[::-1]
            # Compute deep supervision heads
            for i in range(self.deep_supr_num):
                self.heads[i] = self.deep_supervision_heads[i](temp[i+1])
            # Combine deep supervision outputs
            out_all = torch.zeros(out.shape[0], len(self.heads) + 1, *out.shape[1:], device=out.device, dtype=out.dtype)
            out_all[:, 0] = out
            for idx, feature_map in enumerate(self.heads):
                out_all[:, idx + 1] = interpolate(feature_map, out.shape[2:])
            return out_all

        return out

    def forward(self, x, proj_features: bool = False):
        if not proj_features:
            return super(MoonDynUNet, self).forward(x)
        else:
            x, skips = self.encode(x)
            v = self.proj_layer(x)
            out = self.decode(x, skips)
        return out, v

