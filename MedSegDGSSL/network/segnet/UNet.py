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

from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm

__all__ = ["UNet", "Unet"]


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


class UNet(nn.Module):
    """
    Enhanced version of UNet which has residual units implemented with the ResidualUnit class.
    The residual part uses a convolution to change the input dimensions to match the output dimensions
    if this is necessary but will use nn.Identity if not.
    Refer to: https://link.springer.com/chapter/10.1007/978-3-030-12029-0_40.

    Each layer of the network has a encode and decode path with a skip connection between them. Data in the encode path
    is downsampled using strided convolutions (if `strides` is given values greater than 1) and in the decode path
    upsampled using strided transpose convolutions. These down or up sampling operations occur at the beginning of each
    block rather than afterwards as is typical in UNet implementations.

    To further explain this consider the first example network given below. This network has 3 layers with strides
    of 2 for each of the middle layers (the last layer is the bottom connection which does not down/up sample). Input
    data to this network is immediately reduced in the spatial dimensions by a factor of 2 by the first convolution of
    the residual unit defining the first layer of the encode part. The last layer of the decode part will upsample its
    input (data from the previous layer concatenated with data from the skip connection) in the first convolution. this
    ensures the final output of the network has the same shape as the input.

    Padding values for the convolutions are chosen to ensure output sizes are even divisors/multiples of the input
    sizes if the `strides` value for a layer is a factor of the input sizes. A typical case is to use `strides` values
    of 2 and inputs that are multiples of powers of 2. An input can thus be downsampled evenly however many times its
    dimensions can be divided by 2, so for the example network inputs would have to have dimensions that are multiples
    of 4. In the second example network given below the input to the bottom layer will have shape (1, 64, 15, 15) for
    an input of shape (1, 1, 240, 240) demonstrating the input being reduced in size spatially by 2**4.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        channels: sequence of channels. Top block first. The length of `channels` should be no less than 2.
        strides: sequence of convolution strides. The length of `stride` should equal to `len(channels) - 1`.
        kernel_size: convolution kernel size, the value(s) should be odd. If sequence,
            its length should equal to dimensions. Defaults to 3.
        up_kernel_size: upsampling convolution kernel size, the value(s) should be odd. If sequence,
            its length should equal to dimensions. Defaults to 3.
        num_res_units: number of residual units. Defaults to 0.
        act: activation type and arguments. Defaults to PReLU.
        norm: feature normalization type and arguments. Defaults to instance norm.
        dropout: dropout ratio. Defaults to no dropout.
        bias: whether to have a bias term in convolution blocks. Defaults to True.
            According to `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_,
            if a conv layer is directly followed by a batch norm layer, bias should be False.
        adn_ordering: a string representing the ordering of activation (A), normalization (N), and dropout (D).
            Defaults to "NDA". See also: :py:class:`monai.networks.blocks.ADN`.

    Examples::

        from monai.networks.nets import UNet

        # 3 layer network with down/upsampling by a factor of 2 at each layer with 2-convolution residual units
        net = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(4, 8, 16),
            strides=(2, 2),
            num_res_units=2
        )

        # 5 layer network with simple convolution/normalization/dropout/activation blocks defining the layers
        net=UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(4, 8, 16, 32, 64),
            strides=(2, 2, 2, 2),
        )

    .. deprecated:: 0.6.0
        ``dimensions`` is deprecated, use ``spatial_dims`` instead.

    Note: The acceptable spatial size of input data depends on the parameters of the network,
        to set appropriate spatial size, please check the tutorial for more details:
        https://github.com/Project-MONAI/tutorials/blob/master/modules/UNet_input_size_constrains.ipynb.
        Typically, when using a stride of 2 in down / up sampling, the output dimensions are either half of the
        input when downsampling, or twice when upsampling. In this case with N numbers of layers in the network,
        the inputs must have spatial dimensions that are all multiples of 2^N.
        Usually, applying `resize`, `pad` or `crop` transforms can help adjust the spatial size of input data.

    """
    def __init__(
        self,
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


Unet = UNet

@NETWORK_REGISTRY.register()
def naiveunet(model_cfg):
    unet = UNet(spatial_dims=model_cfg.SPATIAL_DIMS,
                in_channels= model_cfg.IN_CHANNELS,
                out_channels= model_cfg.OUT_CHANNELS,
                channels=model_cfg.FEATURES,
                strides=model_cfg.STRIDES,
                norm= model_cfg.NORM,
                dropout = model_cfg.DROPOUT,
                return_features= model_cfg.RETURN_FEATURES)
    return unet

@NETWORK_REGISTRY.register()
def basicunet(model_cfg):
    unet = UNet(spatial_dims=model_cfg.SPATIAL_DIMS,
                in_channels= model_cfg.IN_CHANNELS,
                out_channels= model_cfg.OUT_CHANNELS,
                channels=model_cfg.FEATURES,
                strides=model_cfg.STRIDES,
                num_res_units=2,
                norm= model_cfg.NORM,
                dropout = model_cfg.DROPOUT,
                return_features= model_cfg.RETURN_FEATURES)
    return unet

# Always test your network implementation
if __name__ == '__main__':
    device = torch.device('cuda')
    input = torch.ones((32, 2, 128, 128), device=device)
    network = UNet(spatial_dims=2,
                   in_channels=2, out_channels=3, channels=(32, 64, 96, 128), strides=(2, 2, 2),
                   kernel_size=3, up_kernel_size=3, num_res_units=2,
                   act='relu', norm='BATCH', return_features=True).to(device)
    out, feature = network(input)
    print('Out shape', out.shape,
          'Feature shape', feature.shape)
