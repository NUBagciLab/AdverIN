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
from MedSegDGSSL.network.build import NETWORK_REGISTRY

from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.factories import Norm

__all__ = ["AttentionUnet"]


class AttentionBlock(nn.Module):
    def __init__(self, spatial_dims: int, f_int: int, f_g: int, f_l: int, norm: Norm.BATCH, dropout=0.0):
        super().__init__()
        self.W_g = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=f_g,
                out_channels=f_int,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
                conv_only=True,
            ),
            Norm[norm, spatial_dims](f_int),
        )

        self.W_x = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=f_l,
                out_channels=f_int,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
                conv_only=True,
            ),
            Norm[norm, spatial_dims](f_int),
        )

        self.psi = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=f_int,
                out_channels=1,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
                conv_only=True,
            ),
            Norm[norm, spatial_dims](1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU()

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi: torch.Tensor = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class UpConv(nn.Module):
    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int,
                 strides: int, num_res_units:int,
                 kernel_size: Union[Sequence[int], int] = 3,
                 act: Union[Tuple, str] = Act.PRELU,
                 norm: Union[Tuple, str] = Norm.INSTANCE,
                 dropout: float = 0.0,
                 adn_ordering: str = "NDA",
                 last_conv_only:bool = False):
        super().__init__()
        self.up = Convolution(
            spatial_dims,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=kernel_size,
            act=act,
            adn_ordering=adn_ordering,
            norm=norm,
            dropout=dropout,
            is_transposed=True,
        )
        self.attention = AttentionBlock(spatial_dims=spatial_dims,
                                        f_g=out_channels, f_l=out_channels,
                                        f_int=in_channels // 2, norm=norm)
        self.conv = Convolution(
            spatial_dims,
            2*out_channels,
            out_channels,
            strides=1,
            kernel_size=kernel_size,
            act=act,
            norm=norm,
            dropout=dropout,
            is_transposed=False,
            adn_ordering=adn_ordering,
            conv_only=last_conv_only
        )
        if num_res_units > 0:
            ru = ResidualUnit(
                spatial_dims,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=kernel_size,
                subunits=1,
                act=act,
                norm=norm,
                dropout=dropout,
                adn_ordering=adn_ordering,
            )
            self.conv = nn.Sequential(self.conv, ru)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.up(x)
        skip = self.attention(g=x, x=skip)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    """ Build the encoder for the network
    """
    def __init__(self,
                 spatial_dims: int,
                 in_channels: int,
                 channels: Sequence[int],
                 strides: Sequence[int],
                 kernel_size: Union[Sequence[int], int] = 3,
                 num_res_units: int = 0,
                 act: Union[Tuple, str] = Act.PRELU,
                 norm: Union[Tuple, str] = Norm.INSTANCE,
                 dropout: float = 0.0,
                 bias: bool = True,
                 adn_ordering: str = "NDA",):
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

        self.n_layers = len(channels) -1 
        self.encoder_list = nn.ModuleList()
        self.input_block = self._get_down_layer(self.in_channels, self.channels[0], 1, False)

        for i in range(self.n_layers):
            self.encoder_list.append(self._get_down_layer(in_channels=self.channels[i],
                                                          out_channels=self.channels[i+1],
                                                          strides=self.strides[i], is_top=False))

        self.bottom = self._get_bottom_layer(self.channels[-1], self.channels[-1])

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
    
    def forward(self, input):
        features = []
        input = self.input_block(input)
        for i in range(self.n_layers):
            features.append(input)
            input = self.encoder_list[i](input)
        input = self.bottom(input)
        features.append(input)
        return features


class Decoder(nn.Module):
    """ Build the decoder for the network
    """
    def __init__(self,
                 spatial_dims: int,
                 out_channels: int,
                 channels: Sequence[int],
                 strides: Sequence[int],
                 up_kernel_size: Union[Sequence[int], int] = 3,
                 num_res_units: int = 0,
                 act: Union[Tuple, str] = Act.PRELU,
                 norm: Union[Tuple, str] = Norm.INSTANCE,
                 dropout: float = 0.0,
                 bias: bool = True,
                 adn_ordering: str = "NDA",):
        super().__init__()
        self.dimensions = spatial_dims
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering

        self.n_layers = len(channels) - 1
        self.decoder_list = nn.ModuleList()
        for i in range(self.n_layers):
            self.decoder_list.append(self._get_up_layer(in_channels=self.channels[-i-1],
                                                        out_channels=self.channels[-i-2],
                                                        strides=self.strides[-i-1], is_top=False))
        self.out_block = Convolution(spatial_dims=self.dimensions, 
                                     in_channels=self.channels[0], out_channels=self.out_channels,
                                     strides=1, kernel_size=self.up_kernel_size,
                                     conv_only=True)
    
    
    def _get_up_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
        Returns the decoding (up) part of a layer of the network. This typically will upsample data at some point
        in its structure. Its output is used as input to the next layer up.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        conv: Union[Convolution, nn.Sequential]

        conv = UpConv(spatial_dims=self.dimensions,
                      in_channels=in_channels,
                      out_channels=out_channels,
                      strides=strides,
                      kernel_size=self.up_kernel_size,
                      act=self.act,
                      norm=self.norm,
                      dropout=self.dropout,
                      num_res_units=self.num_res_units,
                      adn_ordering=self.adn_ordering,
                      last_conv_only=is_top)

        return conv

    def forward(self, features):
        out = features[-1]
        for i in range(self.n_layers):
            out = self.decoder_list[i](out, features[-i-2])
        
        out = self.out_block(out)
        return out



class AttentionUNet(nn.Module):
    """
    Attention Unet based on
    Otkay et al. "Attention U-Net: Learning Where to Look for the Pancreas"
    https://arxiv.org/abs/1804.03999

    Args:
        spatial_dims: number of spatial dimensions of the input image.
        in_channels: number of the input channel.
        out_channels: number of the output classes.
        channels (Sequence[int]): sequence of channels. Top block first. The length of `channels` should be no less than 2.
        strides (Sequence[int]): stride to use for convolutions.
        kernel_size: convolution kernel size.
        upsample_kernel_size: convolution kernel size for transposed convolution layers.
        dropout: dropout ratio. Defaults to no dropout.
    """
    def __init__(self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
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

        self.encoder = Encoder(spatial_dims=self.dimensions,
                               in_channels=self.in_channels, channels=self.channels, strides=self.strides,
                               kernel_size=self.kernel_size, num_res_units=self.num_res_units,
                               act=self.act, norm=self.norm, dropout=self.dropout, bias=self.bias,
                               adn_ordering=self.adn_ordering)
        self.decoder = Decoder(spatial_dims=self.dimensions,
                               out_channels=self.out_channels, channels=self.channels, strides=self.strides,
                               up_kernel_size=self.up_kernel_size, num_res_units=self.num_res_units,
                               act=self.act, norm=self.norm, dropout=self.dropout, bias=self.bias,
                               adn_ordering=self.adn_ordering)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        x = self.decoder(features)

        if self.return_features and self.training:
            return x, features[-1]
        else:
            return x

attentionUnet = AttentionUNet

@NETWORK_REGISTRY.register()
def attenionunet(model_cfg):
    attenunet = AttentionUNet(spatial_dims=model_cfg.SPATIAL_DIMS,
                              in_channels= model_cfg.IN_CHANNELS,
                              out_channels= model_cfg.OUT_CHANNELS,
                              features= model_cfg.FEATURES,
                              norm= model_cfg.NORM,
                              dropout = model_cfg.DROPOUT,
                              is_return_feature= model_cfg.RETURN_FEATURES)
    return attenunet

# Always test your network implementation
if __name__ == '__main__':
    device = torch.device('cuda')
    input = torch.ones((32, 2, 128, 128), device=device)
    network = AttentionUNet(spatial_dims=2,
                   in_channels=2, out_channels=3, channels=(32, 64, 96, 128, 256), strides=(2, 2, 2, 2),
                   kernel_size=3, up_kernel_size=3, num_res_units=2,
                   act='relu', norm='BATCH', return_features=True).to(device)
    out, feature = network(input)
    print('Out shape', out.shape,
          'Feature shape', feature.shape)
