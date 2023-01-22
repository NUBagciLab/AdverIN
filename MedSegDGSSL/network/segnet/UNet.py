"""
Modify based on the MONAI implementation
Enable returning the intermidiate features
"""

import warnings
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import SkipConnection
from monai.utils import SkipMode
from monai.networks.nets import UNet

from MedSegDGSSL.network.build import NETWORK_REGISTRY
from MedSegDGSSL.network.ops.utils import Sequential2

__all__ = ["UNet", "Unet"]


class ConvolutionWrapper(Convolution):
    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int,
                 strides: Union[Sequence[int], int] = 1, kernel_size: Union[Sequence[int], int] = 3,
                 adn_ordering: str = "NDA", act: Optional[Union[Tuple, str]] = "PRELU",
                 norm: Optional[Union[Tuple, str]] = "INSTANCE",
                 dropout: Optional[Union[Tuple, str, float]] = None, dropout_dim: Optional[int] = 1,
                 dilation: Union[Sequence[int], int] = 1, groups: int = 1, bias: bool = True,
                 conv_only: bool = False, is_transposed: bool = False,
                 padding: Optional[Union[Sequence[int], int]] = None,
                 output_padding: Optional[Union[Sequence[int], int]] = None,
                 dimensions: Optional[int] = None) -> None:
        super().__init__(spatial_dims, in_channels, out_channels, strides,
                         kernel_size, adn_ordering, act, norm, dropout, dropout_dim,
                         dilation, groups, bias, conv_only, is_transposed, padding,
                         output_padding, dimensions)
    
    def forward(self, input, feature=None):
        input = super().forward(input)
        if feature is None:
            return input, input
        else:
            return input, feature


class ResidualUnitWrapper(ResidualUnit):
    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int,
                 strides: Union[Sequence[int], int] = 1, kernel_size: Union[Sequence[int], int] = 3,
                 subunits: int = 2, adn_ordering: str = "NDA", act: Optional[Union[Tuple, str]] = "PRELU",
                 norm: Optional[Union[Tuple, str]] = "INSTANCE", dropout: Optional[Union[Tuple, str, float]] = None,
                 dropout_dim: Optional[int] = 1, dilation: Union[Sequence[int], int] = 1, bias: bool = True,
                 last_conv_only: bool = False, padding: Optional[Union[Sequence[int], int]] = None,
                 dimensions: Optional[int] = None) -> None:
        super().__init__(spatial_dims, in_channels, out_channels, strides, kernel_size,
                         subunits, adn_ordering, act, norm, dropout, dropout_dim, dilation,
                         bias, last_conv_only, padding, dimensions)
    
    def forward(self, input, feature=None):
        input = super().forward(input)
        if feature is None:
            return input, input
        else:
            return input, feature


class SkipConnectionWrapper(SkipConnection):
    def __init__(self, submodule, dim: int = 1, mode: Union[str, SkipMode] = "cat") -> None:
        super().__init__(submodule, dim, mode)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, feature = self.submodule(x)

        if self.mode == "cat":
            return torch.cat([x, y], dim=self.dim), feature
        if self.mode == "add":
            return torch.add(x, y), feature
        if self.mode == "mul":
            return torch.mul(x, y), feature
        raise NotImplementedError(f"Unsupported mode {self.mode}.")


class UNet(nn.Module):
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

        def _create_block(
            inc: int, outc: int, channels: Sequence[int], strides: Sequence[int], is_top: bool
        ) -> nn.Module:
            """
            Builds the UNet structure from the bottom up by recursing down to the bottom block, then creating sequential
            blocks containing the downsample path, a skip connection around the previous block, and the upsample path.

            Args:
                inc: number of input channels.
                outc: number of output channels.
                channels: sequence of channels. Top block first.
                strides: convolution stride.
                is_top: True if this is the top block.
            """
            c = channels[0]
            s = strides[0]

            subblock: nn.Module

            if len(channels) > 2:
                subblock = _create_block(c, c, channels[1:], strides[1:], False)  # continue recursion down
                upc = c * 2
            else:
                # the next layer is the bottom so stop recursion, create the bottom layer as the sublock for this layer
                subblock = self._get_bottom_layer(c, channels[1])
                upc = c + channels[1]

            down = self._get_down_layer(inc, c, s, is_top)  # create layer in downsampling path
            up = self._get_up_layer(upc, outc, s, is_top)  # create layer in upsampling path

            return self._get_connection_block(down, up, subblock)

        self.input_block = Convolution(self.dimensions, in_channels, self.channels[0],
                                       strides=1, kernel_size=self.kernel_size, act=self.act,
                                       norm=self.norm, dropout=self.dropout, bias=self.bias, adn_ordering=self.adn_ordering)
        self.output_block = Convolution(self.dimensions, self.channels[0], out_channels,
                                        strides=1, kernel_size=self.kernel_size, act=self.act, conv_only=True,
                                        norm=self.norm, dropout=self.dropout, bias=self.bias, adn_ordering=self.adn_ordering)
        
        self.inter_model = _create_block(self.channels[0], self.channels[0], self.channels, self.strides, False)
        self.model = nn.Sequential(self.input_block, self.inter_model, self.output_block)

    def _get_connection_block(self, down_path: nn.Module, up_path: nn.Module, subblock: nn.Module) -> nn.Module:
        """
        Returns the block object defining a layer of the UNet structure including the implementation of the skip
        between encoding (down) and and decoding (up) sides of the network.

        Args:
            down_path: encoding half of the layer
            up_path: decoding half of the layer
            subblock: block defining the next layer in the network.
        Returns: block for this layer: `nn.Sequential(down_path, SkipConnection(subblock), up_path)`
        """
        return nn.Sequential(down_path, SkipConnection(subblock), up_path)

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

        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=True,
            adn_ordering=self.adn_ordering,
        )

        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_top,
                adn_ordering=self.adn_ordering,
            )
            conv = nn.Sequential(conv, ru)

        return conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x


class UNetWithFeature(nn.Module):
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
        self.features = {}


        def _create_block(
            inc: int, outc: int, channels: Sequence[int], strides: Sequence[int], is_top: bool
        ) -> nn.Module:
            """
            Builds the UNet structure from the bottom up by recursing down to the bottom block, then creating sequential
            blocks containing the downsample path, a skip connection around the previous block, and the upsample path.

            Args:
                inc: number of input channels.
                outc: number of output channels.
                channels: sequence of channels. Top block first.
                strides: convolution stride.
                is_top: True if this is the top block.
            """
            c = channels[0]
            s = strides[0]

            subblock: nn.Module

            if len(channels) > 2:
                subblock = _create_block(c, c, channels[1:], strides[1:], False)  # continue recursion down
                upc = c * 2
            else:
                # the next layer is the bottom so stop recursion, create the bottom layer as the sublock for this layer
                subblock = self._get_bottom_layer(c, channels[1])
                upc = c + channels[1]

            down = self._get_down_layer(inc, c, s, is_top)  # create layer in downsampling path
            up = self._get_up_layer(upc, outc, s, is_top)  # create layer in upsampling path
            up.register_forward_hook(self.forward_hook(str(outc)))

            return self._get_connection_block(down, up, subblock)

        self.input_block = Convolution(self.dimensions, in_channels, self.channels[0],
                                       strides=1, kernel_size=self.kernel_size, act=self.act,
                                       norm=self.norm, dropout=self.dropout, bias=self.bias, adn_ordering=self.adn_ordering)
        self.output_block = ConvolutionWrapper(self.dimensions, self.channels[0], out_channels,
                                               strides=1, kernel_size=self.kernel_size, act=self.act, conv_only=True,
                                               norm=self.norm, dropout=self.dropout, bias=self.bias, adn_ordering=self.adn_ordering)
        
        self.inter_model = _create_block(self.channels[0], self.channels[0], self.channels, self.strides, False)
        self.model = Sequential2(self.input_block, self.inter_model, self.output_block)

    def forward_hook(self, layer_name):
        def hook(module, input, output):
            self.features[layer_name] = output
        return hook

    def _get_connection_block(self, down_path: nn.Module, up_path: nn.Module, subblock: nn.Module) -> nn.Module:
        """
        Returns the block object defining a layer of the UNet structure including the implementation of the skip
        between encoding (down) and and decoding (up) sides of the network.

        Args:
            down_path: encoding half of the layer
            up_path: decoding half of the layer
            subblock: block defining the next layer in the network.
        Returns: block for this layer: `nn.Sequential(down_path, SkipConnection(subblock), up_path)`
        """
        return Sequential2(down_path, SkipConnectionWrapper(subblock), up_path)

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
        mod: nn.Module
        if self.num_res_units > 0:

            mod = ResidualUnitWrapper(
                self.dimensions,
                in_channels,
                out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                adn_ordering=self.adn_ordering,
            )
            return mod
        mod = ConvolutionWrapper(
            self.dimensions,
            in_channels,
            out_channels,
            strides=1,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
        )

        return mod

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
        conv: Union[Convolution, nn.Sequential, Sequential2]

        conv = ConvolutionWrapper(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=True,
            adn_ordering=self.adn_ordering,
        )

        if self.num_res_units > 0:
            ru = ResidualUnitWrapper(
                self.dimensions,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_top,
                adn_ordering=self.adn_ordering,
            )
            conv = Sequential2(conv, ru)
        
        return conv

    def forward(self, x: torch.Tensor):

        x, feature = self.model(x)
        if self.training:
            return x, feature
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
                dropout = model_cfg.DROPOUT)
    return unet

@NETWORK_REGISTRY.register()
def basicunet(model_cfg):
    unet = UNet(spatial_dims=model_cfg.SPATIAL_DIMS,
                in_channels= model_cfg.IN_CHANNELS,
                out_channels= model_cfg.OUT_CHANNELS,
                channels=model_cfg.FEATURES,
                strides=model_cfg.STRIDES,
                num_res_units=1,
                norm= model_cfg.NORM,
                dropout = model_cfg.DROPOUT)
    return unet

@NETWORK_REGISTRY.register()
def basicunet_withfeature(model_cfg):
    unet = UNetWithFeature(spatial_dims=model_cfg.SPATIAL_DIMS,
                in_channels= model_cfg.IN_CHANNELS,
                out_channels= model_cfg.OUT_CHANNELS,
                channels=model_cfg.FEATURES,
                strides=model_cfg.STRIDES,
                num_res_units=1,
                norm= model_cfg.NORM,
                dropout = model_cfg.DROPOUT)
    return unet

# Always test your network implementation
if __name__ == '__main__':
    device = torch.device('cuda')
    input = torch.ones((32, 2, 128, 128), device=device)
    network = UNet(spatial_dims=2,
                   in_channels=2, out_channels=3, channels=(32, 64, 96, 128), strides=(2, 2, 2),
                   kernel_size=3, up_kernel_size=3, num_res_units=1,
                   act='relu', norm='BATCH').to(device)
    out, feature = network(input)
    print('Out shape', out.shape,
          'Feature shape', feature.shape)
