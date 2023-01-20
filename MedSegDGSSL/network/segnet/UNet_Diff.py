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

import pdb
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat

from MedSegDGSSL.network.build import NETWORK_REGISTRY

from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm

__all__ = ["UNet_EncDec", "Unet_EncDec"]


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding


class ConvolutionWithTime(Convolution):
    def __init__(self, spatial_dims: int, in_channels: int, embed_channels:int,
                 out_channels: int, strides: Union[Sequence[int], int] = 1,
                 kernel_size: Union[Sequence[int], int] = 3, adn_ordering: str = "NDA",
                 act: Optional[Union[Tuple, str]] = "PRELU", norm: Optional[Union[Tuple, str]] = "INSTANCE", dropout: Optional[Union[Tuple, str, float]] = None, dropout_dim: Optional[int] = 1, dilation: Union[Sequence[int], int] = 1, groups: int = 1, bias: bool = True, conv_only: bool = False, is_transposed: bool = False, padding: Optional[Union[Sequence[int], int]] = None, output_padding: Optional[Union[Sequence[int], int]] = None, dimensions: Optional[int] = None) -> None:

        super().__init__(spatial_dims, in_channels,
                         out_channels, strides, kernel_size, adn_ordering,
                         act, norm, dropout, dropout_dim, dilation,
                         groups, bias, conv_only, is_transposed, padding,
                         output_padding, dimensions)
        self.time_embed_dim = embed_channels // 2
        self.time_embed = nn.Sequential(*[nn.Linear(self.time_embed_dim, embed_channels),
                                          nn.SiLU(),
                                          nn.Linear(embed_channels, out_channels)])
    
    def forward(self, x, t):
        x = super().forward(x)
        t = timestep_embedding(t, dim=self.time_embed_dim)
        t = self.time_embed(t)
        t = torch.reshape(t, (*t.shape, *((1,) * (x.dim()-t.dim()))))
        return x + t


class ResidualUnitWithTime(ResidualUnit):
    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int, embed_channels: int,
                 strides: Union[Sequence[int], int] = 1, kernel_size: Union[Sequence[int], int] = 3,
                 subunits: int = 2, adn_ordering: str = "NDA", act: Optional[Union[Tuple, str]] = "PRELU",
                 norm: Optional[Union[Tuple, str]] = "INSTANCE", dropout: Optional[Union[Tuple, str, float]] = None,
                 dropout_dim: Optional[int] = 1, dilation: Union[Sequence[int], int] = 1, bias: bool = True, last_conv_only: bool = False, padding: Optional[Union[Sequence[int], int]] = None, dimensions: Optional[int] = None) -> None:
        super().__init__(spatial_dims, in_channels, out_channels, strides,
                         kernel_size, subunits, adn_ordering, act, norm, dropout,
                         dropout_dim, dilation, bias, last_conv_only, padding, dimensions)
        self.time_embed_dim = embed_channels // 2
        self.time_embed = nn.Sequential(*[nn.Linear(self.time_embed_dim, embed_channels),
                                          nn.SiLU(),
                                          nn.Linear(embed_channels, out_channels)])

    def forward(self, x, t):
        x = super().forward(x)
        t = timestep_embedding(t, dim=self.time_embed_dim)
        t = self.time_embed(t)
        t = torch.reshape(t, (*t.shape, *((1,) * (x.dim()-t.dim()))))
        return x + t


class ExtractContext(nn.Module):
    def __init__(self, timesteps, num_stages):
        super().__init__()
        self.timesteps = timesteps
        self.num_stages = num_stages

        self.register_buffer('stage_weight',
                             torch.from_numpy(ExtractContext.generate_stage_weights(timesteps, num_stages)).to(torch.float))

    def forward(self, context_list, t):
        context = torch.stack(context_list, dim=1)
        
        weight = ExtractContext.extract(self.stage_weight, t, context.shape)
        context = torch.sum(context*weight, dim=1, keepdim=False)
        return context

    @staticmethod
    def generate_stage_weights(timesteps, num_stages):
        stage_weights = np.zeros(shape=(timesteps, num_stages))
        period = timesteps // num_stages

        for i in range(num_stages):
            if i != (num_stages-1):
                stage_weights[i*period:(i+1)*period, i] = (np.cos(np.pi*np.linspace(0, 1, period))+1) / 2
                stage_weights[i*period:(i+1)*period, i+1] = 1 - stage_weights[i*period:(i+1)*period, i]
            else:
                stage_weights[i*period:, -1] = 1.

        return stage_weights
    
    @staticmethod
    def extract(a, t, x_shape):
        num_stages = a.shape[1]
        b = t.shape[0]
        out = torch.index_select(a, dim=0, index=t.to(torch.long))
        return out.reshape(b, num_stages, *((1,) * (len(x_shape)-2)))


class CrossAttentionCondition(nn.Module):
    """ Generate the conditioned noise using the extracted features
    """
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context):
        h = self.heads

        # print(x.dtype, context.dtype)
        # pdb.set_trace()
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class SimpleConcate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x, context):
        return torch.cat([x, context], dim=1)

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


class contextEncoder(nn.Module):
    """ Build the encoder for the network
    """
    def __init__(self,
                 spatial_dims: int,
                 in_channels: int,
                 channels: Sequence[int],
                 project_channels: int,
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
        self.project_channels = project_channels
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
        self.project_list = nn.ModuleList()
        self.upsample_list = nn.ModuleList()

        self.input_block = self._get_down_layer(self.in_channels, self.channels[0], 1, False)
        self.project_list.append(nn.Linear(in_features=self.channels[0],
                                           out_features=self.project_channels))
        self.upsample_list.append(nn.Identity())
    
        for i in range(self.n_layers):
            self.encoder_list.append(self._get_down_layer(in_channels=self.channels[i],
                                                          out_channels=self.channels[i+1],
                                                          strides=self.strides[i], is_top=False))
            self.project_list.append(nn.Linear(in_features=self.channels[i+1],
                                               out_features=self.project_channels))
            self.upsample_list.append(nn.Upsample(scale_factor=int(np.prod(self.strides[:(i+1)]))))

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
            feature = self.upsample_list[i](input)
            b, c, *_hwd = feature.shape
            feature = torch.transpose(torch.flatten(feature, start_dim=2), 1, 2)
            feature = self.project_list[i](feature)
            feature = torch.reshape(torch.transpose(feature, 1, 2), (b, -1, *_hwd))
            features.append(feature)
            input = self.encoder_list[i](input)

        feature = self.upsample_list[-1](self.bottom(input))
        b, c, *_hwd = feature.shape
        feature = torch.transpose(torch.flatten(feature, start_dim=2), 1, 2)
        feature = self.project_list[-1](feature)
        feature = torch.reshape(torch.transpose(feature, 1, 2), (b, -1, *_hwd))
        features.append(feature)

        return features


class Encoder(nn.Module):
    """ Build the encoder for the network
    """
    def __init__(self,
                 spatial_dims: int,
                 in_channels: int,
                 channels: Sequence[int],
                 embed_channel: int,
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
        self.embed_channel = embed_channel
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

            mod = ResidualUnitWithTime(
                self.dimensions,
                in_channels,
                out_channels,
                self.embed_channel,
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
        mod = ConvolutionWithTime(
            self.dimensions,
            in_channels,
            out_channels,
            self.embed_channel,
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
    
    def forward(self, input, t):
        features = []
        input = self.input_block(input, t)
        for i in range(self.n_layers):
            features.append(input)
            input = self.encoder_list[i](input, t)
        input = self.bottom(input, t)
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



class UNet_EncDec(nn.Module):
    """
    Encoder-Decoder Style UNet

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
        embed_channel: int, 
        time_steps:int, 
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
        self.embed_channel = embed_channel
        self.time_steps = time_steps
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
        
        self.context_encoder = contextEncoder(spatial_dims=self.dimensions,
                                              in_channels=self.in_channels, channels=self.channels,
                                              strides=self.strides, project_channels=self.embed_channel//2,
                                              kernel_size=self.kernel_size, num_res_units=self.num_res_units,
                                              act=self.act, norm=self.norm, dropout=self.dropout, bias=self.bias,
                                              adn_ordering=self.adn_ordering)
        self.context_extract = ExtractContext(self.time_steps,
                                              num_stages=len(self.channels))
        self.y_project = nn.Linear(in_features=self.out_channels, out_features=self.embed_channel//2)
        '''self.cross_attention = CrossAttentionCondition(query_dim=self.embed_channel,
                                                       context_dim=self.embed_channel,
                                                       heads=8, dim_head=32)'''

        self.encoder = Encoder(spatial_dims=self.dimensions,
                               in_channels=self.embed_channel, channels=self.channels,
                               embed_channel=embed_channel, strides=self.strides,
                               kernel_size=self.kernel_size, num_res_units=self.num_res_units,
                               act=self.act, norm=self.norm, dropout=self.dropout, bias=self.bias,
                               adn_ordering=self.adn_ordering)
        self.decoder = Decoder(spatial_dims=self.dimensions,
                               out_channels=self.out_channels, channels=self.channels, strides=self.strides,
                               up_kernel_size=self.up_kernel_size, num_res_units=self.num_res_units,
                               act=self.act, norm=self.norm, dropout=self.dropout, bias=self.bias,
                               adn_ordering=self.adn_ordering)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, with_context:bool=True) -> torch.Tensor:
        ### Encode the features into y using conditioning
        b, c, *_hwd = y.shape
        y = torch.transpose(torch.flatten(y, start_dim=2), 1, 2)
        # pdb.set_trace()
        y = self.y_project(y)

        if with_context:
            
            context = self.context_extract(self.context_encoder(x), t)
            context = torch.transpose(torch.flatten(context, start_dim=2), 1, 2)
            # y = self.cross_attention(y, context)
            y = torch.cat([y, context], dim=1)
        else:
            y = torch.cat([y, y], dim=1)

        y = torch.reshape(torch.transpose(y, 1, 2), (b, -1, *_hwd))
        features = self.encoder(y, t)
        y = self.decoder(features)

        return y

Unet_EncDec = UNet_EncDec

@NETWORK_REGISTRY.register()
def diff_unet(model_cfg):
    unet = UNet_EncDec(spatial_dims=model_cfg.SPATIAL_DIMS,
                       in_channels= model_cfg.IN_CHANNELS,
                       out_channels= model_cfg.OUT_CHANNELS,
                       channels=model_cfg.FEATURES,
                       embed_channel=model_cfg.EMBED_CHANNEL,
                       time_steps=model_cfg.TIME_STEPS,
                       strides=model_cfg.STRIDES,
                       num_res_units=1,
                       norm= model_cfg.NORM,
                       dropout = model_cfg.DROPOUT)
    return unet


# Always test your network implementation
if __name__ == '__main__':
    device = torch.device('cuda')
    batch_size = 32
    input = torch.ones((batch_size, 3, 384, 384), device=device).to(torch.float32)
    noise = torch.randn((batch_size, 3, 384, 384), device=device).to(torch.float32)
    t = torch.randint(low=0, high=400, size=(batch_size,), device=device)
    network = UNet_EncDec(spatial_dims=2, embed_channel=32, time_steps=400, 
                          in_channels=3, out_channels=3, channels=(32, 64, 96, 128), strides=(2, 2, 2),
                          kernel_size=3, up_kernel_size=3, num_res_units=2,
                          act='relu', norm='BATCH', return_features=True).to(device)
    out = network(x=input, y=noise, t=t)
    print('Out shape', out.shape)
