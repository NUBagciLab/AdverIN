"""
Modify based on the MONAI implementation
Enable returning the intermidiate features
"""

# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import warnings
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from MedSegDGSSL.network.segnet.build import NETWORK_REGISTRY
from MedSegDGSSL.network.segnet.UNet import UpConv, Decoder
from MedSegDGSSL.network.ops.style_augmentation import DSBN

from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm

__all__ = ["UNet_domainops", "Unet_domainops"]


class DomainEncoder(nn.Module):
    """ Build the encoder for the network with style augmentation
    """
    def __init__(self,
                 spatial_dims: int,
                 in_channels: int,
                 channels: Sequence[int],
                 strides: Sequence[int],
                 domain_ops: nn.Module,
                 domainops_kwargs:dict = {},
                 kernel_size: Union[Sequence[int], int] = 3,
                 num_res_units: int = 0,
                 act: Union[Tuple, str] = Act.PRELU,
                 norm: Union[Tuple, str] = Norm.INSTANCE,
                 dropout: float = 0.0,
                 bias: bool = True,
                 adn_ordering: str = "NDA",
                 ):
        super().__init__()
        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering
        self.domain_ops = domain_ops
        self.domainops_kwargs = domainops_kwargs

        self.n_layers = len(channels) -1 
        self.encoder_list = nn.ModuleList()
        self.domainops_list = nn.ModuleList()
        self.input_block = self._get_down_layer(self.in_channels, self.channels[0], 1, False)

        for i in range(self.n_layers):
            self.domainops_list.append(self.domain_ops(num_features=self.channels[-1], **self.domainops_kwargs))
            self.encoder_list.append(self._get_down_layer(in_channels=self.channels[i],
                                                          out_channels=self.channels[i+1],
                                                          strides=self.strides[i], is_top=False))

        self.bottom = self._get_bottom_layer(self.channels[-1], self.channels[-1])
        self.domainops_list.append(self.domain_ops(num_features=self.channels[-1], **self.domainops_kwargs))

    def _get_down_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
        Returns the encoding (down) part of a layer of the network. This typically will downsample data at some point
        in its structure. Its output is used as input to the next layer down and is concatenated with output from the
        next layer to form the input for the decode (up) part of the layer.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        mod: nn.Module
        if self.num_res_units > 0:

            mod = ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                adn_ordering=self.adn_ordering,
            )
            return mod
        mod = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
        )
        return mod

    def _get_bottom_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        """
        Returns the bottom or bottleneck layer at the bottom of the network linking encode to decode halves.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
        """
        return self._get_down_layer(in_channels, out_channels, 1, False)
    
    def forward(self, input, domains):
        features = []
        input = self.input_block(input)
        for i in range(self.n_layers):
            input = self.domainops_list[i](input, domains)
            features.append(input)
            input = self.encoder_list[i](input)

        input = self.domainops_list[-1](input)
        input = self.bottom(input)
        features.append(input)
        return features


class DomainUNet(nn.Module):
    """ Basic UNet Segmentation with domain specific information,
    including BatchInstanceNorm, Domain Specific BatchNorm
    """
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        domainops: nn.Module,
        domainops_kwargs:dict = {},
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
        dimensions: Optional[int] = None,
        return_features:bool = False,
    ) -> None:

        super().__init__()

        if len(channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        delta = len(strides) - (len(channels) - 1)
        if delta < 0:
            raise ValueError("the length of `strides` should equal to `len(channels) - 1`.")
        if delta > 0:
            warnings.warn(f"`len(strides) > len(channels) - 1`, the last {delta} values of strides will not be used.")
        if dimensions is not None:
            spatial_dims = dimensions
        if isinstance(kernel_size, Sequence):
            if len(kernel_size) != spatial_dims:
                raise ValueError("the length of `kernel_size` should equal to `dimensions`.")
        if isinstance(up_kernel_size, Sequence):
            if len(up_kernel_size) != spatial_dims:
                raise ValueError("the length of `up_kernel_size` should equal to `dimensions`.")

        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering
        self.return_features = return_features
        self.domain_ops = domainops
        self.domainops_kwargs = domainops_kwargs

        self.encoder = DomainEncoder(spatial_dims=self.dimensions,
                                        in_channels=self.in_channels, channels=self.channels, strides=self.strides,
                                        domainops=self.domain_ops, domainops_kwargs=self.domainops_kwargs,
                                        kernel_size=self.kernel_size, num_res_units=self.num_res_units,
                                        act=self.act, norm=self.norm, dropout=self.dropout, bias=self.bias,
                                        adn_ordering=self.adn_ordering)
        self.decoder = Decoder(spatial_dims=self.dimensions,
                               out_channels=self.out_channels, channels=self.channels, strides=self.strides,
                               up_kernel_size=self.up_kernel_size, num_res_units=self.num_res_units,
                               act=self.act, norm=self.norm, dropout=self.dropout, bias=self.bias,
                               adn_ordering=self.adn_ordering)
    
    def forward(self, x: torch.Tensor, domain_idx) -> torch.Tensor:
        features = self.encoder(x, domain_idx)
        x = self.decoder(features)

        if self.return_features and self.training:
            return x, features[-1]
        else:
            return x
    
    def forward(self, x: torch.Tensor, domain_idx=None) -> torch.Tensor:
        if domain_idx is not None:
            features = self.encoder(x, self.domains[domain_idx])
            x = self.decoder(features)

            if self.return_features and self.training:
                return x, features[-1]
            else:
                return x
        else:
            out = []
            for domain_idx in range(self.num_domains):
                features = self.encoder(x, self.domains[domain_idx])
                out.append(self.decoder(features))

            return torch.mean(torch.stack(out, dim=0), dim=0, keepdim=False)


DomainUnet = DomainUNet


# Always test your network implementation
if __name__ == '__main__':
    device = torch.device('cuda')
    input = torch.ones((32, 2, 128, 128), device=device)
    network = DomainUNet(spatial_dims=2,
                         in_channels=2, out_channels=3, channels=(32, 64, 96, 128), strides=(2, 2, 2),
                         kernel_size=3, up_kernel_size=3, num_res_units=2,
                         act='relu', norm='BATCH', return_features=True).to(device)
    out, feature = network(input)
    print('Out shape', out.shape,
          'Feature shape', feature.shape)
