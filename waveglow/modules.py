# Copyright 2018 ASLP@NPU.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: npuichigo@gmail.com (zhangyuchao)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import torch
import torch.nn as nn

from torch.nn import functional as F
from torch.nn import Parameter


class SqueezeLayer(nn.Module):
    def __init__(self, factor):
        super(SqueezeLayer, self).__init__()
        self.factor = factor

    def forward(self, input, logdet=None, reverse=False, **kwargs):
        if not reverse:
            assert input.size(-1) % self.factor == 0
            output = input.view(input.size(0), input.size(1), -1, self.factor)
            output = output.permute(0, 1, 3, 2).contiguous()
            output = output.view(input.size(0), -1, input.size(-1) // self.factor)
            return output, logdet
        else:
            assert input.size(1) % self.factor == 0
            output = input.view(input.size(0), -1, self.factor, input.size(-1)) 
            output = output.permute(0, 1, 3, 2).contiguous()
            output = output.view(input.size(0), input.size(1) // self.factor, -1)
            return output, logdet


class InvertibleConv1d(nn.Module):
    def __init__(self, channels):
        super(InvertibleConv1d, self).__init__()
        w_shape = [channels, channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        # Sample a random orthogonal matrix:
        self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        self.w_shape = w_shape

    def get_weight(self, input, reverse):
        w_shape = self.w_shape
        dlogdet = torch.log(torch.abs(torch.det(self.weight))) * input.size(-1)
        if not reverse:
            weight = self.weight.view(w_shape[0], w_shape[1], 1)
        else:
            weight = torch.inverse(self.weight).view(w_shape[0], w_shape[1], 1)
        return weight, dlogdet

    def forward(self, input, logdet=None, reverse=False, **kwargs):
        weight, dlogdet = self.get_weight(input, reverse)
        if not reverse:
            output = F.conv1d(input, weight)
            if logdet is not None:
                logdet = logdet + dlogdet
            return output, logdet
        else:
            output = F.conv1d(input, weight)
            if logdet is not None:
                logdet = logdet - dlogdet
            return output, logdet

    def extra_repr(self):
        return '{}, {}, kernel_size={}, stride={}'.format(
            self.w_shape[0], self.w_shape[1], (1,), (1,))


class AffineCouplingLayer(nn.Module):
    def __init__(self, input_channels, wn_filter_width, wn_dilation_layers,
                 wn_residual_channels, wn_dilation_channels, wn_skip_channels,
                 local_condition_channels):
        super(AffineCouplingLayer, self).__init__()

        self.input_channels = input_channels

        self.wavenet = WaveNet(filter_width=wn_filter_width,
                               dilations=[2 ** i for i in range(wn_dilation_layers)],
                               residual_channels=wn_residual_channels,
                               dilation_channels=wn_dilation_channels,
                               skip_channels=wn_skip_channels,
                               input_channels=input_channels // 2,
                               output_channels=input_channels,
                               local_condition_channels=local_condition_channels)

    def forward(self, input, logdet=None, reverse=False, local_condition=None):
        if not reverse:
            x_a, x_b = torch.split(input, self.input_channels // 2, 1)
            log_s, t = torch.split(
                self.wavenet(x_a, local_condition), self.input_channels // 2, 1)
            x_b = torch.exp(log_s) * x_b + t
            output = torch.cat([x_a, x_b], 1)
            if logdet is not None:
                logdet = logdet + torch.sum(log_s, (1, 2)) 
            return output, logdet
        else:
            x_a, x_b = torch.split(input, self.input_channels // 2, 1)
            log_s, t = torch.split(
                self.wavenet(x_a, local_condition), self.input_channels // 2, 1)
            x_b = (x_b - t) * torch.exp(-log_s)
            output = torch.cat([x_a, x_b], 1)
            if logdet is not None:
                logdet = logdet - torch.sum(log_s, (1, 2))
            return output, logdet


class FlowStep(nn.Module):
    def __init__(self, input_channels, wn_filter_width, wn_dilation_layers,
                 wn_residual_channels, wn_dilation_channels, wn_skip_channels,
                 local_condition_channels):
        super(FlowStep, self).__init__()

        self.input_channels = input_channels

        self.layers = nn.ModuleList()

        self.layers.extend([
            InvertibleConv1d(input_channels),
            AffineCouplingLayer(input_channels, wn_filter_width,
                                wn_dilation_layers, wn_residual_channels,
                                wn_dilation_channels, wn_skip_channels,
                                local_condition_channels)])

    def forward(self, input, logdet=None, reverse=False, local_condition=None):
        if not reverse:
            output = input
            for layer in self.layers:
                output, logdet = layer(output, logdet=logdet, reverse=reverse,
                                       local_condition=local_condition)
            return output, logdet
        else:
            output = input
            for layer in reversed(self.layers):
                output, logdet = layer(output, logdet=logdet, reverse=reverse,
                                       local_condition=local_condition)
            return output, logdet


class GatedDilatedConv1d(nn.Module):
    """Creates a single causal dilated convolution layer.

    The layer contains a gated filter that connects to dense output
    and to a skip connection:
           |-> [gate]   -|        |-> 1x1 conv -> skip output
           |             |-> (*) -|
    input -|-> [filter] -|        |-> 1x1 conv -|
           |                                    |-> (+) -> dense output
           |------------------------------------|
    Where `[gate]` and `[filter]` are causal convolutions with a
    non-linear activation at the output. Biases and global conditioning
    are omitted due to the limits of ASCII art.
    """

    def __init__(self,
                 filter_width,
                 dilation,
                 residual_channels,
                 dilation_channels,
                 skip_channels,
                 local_condition_channels=None):
        """Initializes the GatedDilatedConv1d.

        Args:
            filter_width:
            dilation:
            residual_channels:
            dilation_channels:
            skip_channels:
            local_condition_channels:
        """
        super(GatedDilatedConv1d, self).__init__()
        self.filter_width = filter_width
        self.dilation = dilation
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.skip_channels = skip_channels
        self.local_condition_channels = local_condition_channels

        if filter_width % 2 == 0:
            raise ValueError("You must specify an odd number to filter_width "
                             "to make sure the shape is invariant after conv.")
        padding = (filter_width - 1) // 2 * dilation
        self.conv_filter_gate = nn.Conv1d(residual_channels,
                                          dilation_channels * 2,
                                          filter_width, padding=padding,
                                          dilation=dilation)

        if local_condition_channels is not None:
            self.conv_lc_filter_gate = nn.Conv1d(local_condition_channels,
                                                 dilation_channels * 2, 1,
                                                 bias=False)

        # The 1x1 conv to produce the residual output
        self.conv_dense = nn.Conv1d(dilation_channels, residual_channels, 1)

        # The 1x1 conv to produce the skip output
        self.conv_skip = nn.Conv1d(dilation_channels, skip_channels, 1)

    def forward(self, sample, local_condition):
        """
        Args:
            sample: Shape: [batch_size, channels, time].
            local_condition: Shape: [batch_size, channels, time].
        """
        sample_filter_gate = self.conv_filter_gate(sample)

        if self.local_condition_channels is not None:
            lc_filter_gate = self.conv_lc_filter_gate(local_condition)
            sample_filter_gate += lc_filter_gate

        sample_filter, sample_gate = torch.split(
            sample_filter_gate, self.dilation_channels, 1)
        gated_sample_batch = torch.tanh(sample_filter) * torch.sigmoid(sample_gate)

        # The 1x1 conv to produce the residual output
        transformed = self.conv_dense(gated_sample_batch)
        residual_output = transformed + sample

        # The 1x1 conv to produce the skip output
        skip_output = self.conv_skip(gated_sample_batch)
        return residual_output, skip_output


class WaveNet(nn.Module):
    """Implements the WaveNet block for waveglow.

    Usage (with the architecture as in the DeepMind paper):
        dilations = [2**i for i in range(N)] * M
        filter_width = 2  # Convolutions just use 2 samples.
        residual_channels = 16  # Not specified in the paper.
        dilation_channels = 32  # Not specified in the paper.
        skip_channels = 16      # Not specified in the paper.
        net = WaveNet(dilations, filter_width,
                           residual_channels, dilation_channels,
                           skip_channels)
        output_batch = net(input_batch)
    """

    def __init__(self,
                 filter_width,
                 dilations,
                 residual_channels,
                 dilation_channels,
                 skip_channels,
                 input_channels,
                 output_channels,
                 local_condition_channels=None):
        """Initializes the WaveNet model.

        Args:
            filter_width: The samples that are included in each convolution,
                after dilating.
            dilations: A list with the dilation factor for each layer.
            residual_channels: How many filters to learn for the residual.
            dilation_channels: How many filters to learn for the dilated
                convolution.
            skip_channels: How many filters to learn that contribute to the
                quantized softmax output.
            input_channels:
            output_channels:
            local_condition_channels: Number of channels in local conditioning
                vector. None indicates there is no local conditioning.
        """
        super(WaveNet, self).__init__()
        self.filter_width = filter_width
        self.dilations = dilations
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.skip_channels = skip_channels
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.local_condition_channels = local_condition_channels

        # Creates the conv layer for channel transformation.
        self.preprocessing_layer = nn.Conv1d(input_channels,
                                             residual_channels, 1)

        # Creates the dilated conv layers.
        self.dilated_conv_layers = nn.ModuleList()
        for dilation in dilations:
            conv = GatedDilatedConv1d(
                filter_width=filter_width,
                dilation=dilation,
                residual_channels=residual_channels,
                dilation_channels=dilation_channels,
                skip_channels=skip_channels,
                local_condition_channels=local_condition_channels)
            self.dilated_conv_layers.append(conv)

        # Performs (+) -> ReLU -> 1x1 conv -> ReLU -> 1x1 conv to
        # postprocess the output.
        self.postprocessing_layers = nn.ModuleList([
            nn.ReLU(inplace=True),
            nn.Conv1d(skip_channels, skip_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(skip_channels, self.output_channels, 1)
        ])

    def forward(self, sample, local_condition):
        current_layer = self.preprocessing_layer(sample)

        skip_outputs = []
        for dilated_conv_layer in self.dilated_conv_layers:
            current_layer, skip_output = dilated_conv_layer(current_layer,
                                                            local_condition)
            skip_outputs.append(skip_output)

        # Adds up skip connections from the outputs of each layer.
        current_layer = sum(skip_outputs)

        for postprocessing_layer in self.postprocessing_layers:
            current_layer = postprocessing_layer(current_layer)

        return current_layer


class UpsampleNet(nn.Module):

    def __init__(self, upsample_factor, upsample_method='duplicate', squeeze_factor=8):
        super(UpsampleNet, self).__init__()
        self.upsample_factor = upsample_factor
        self.upsample_method = upsample_method
        self.squeeze_factor = squeeze_factor 

        if upsample_method == 'duplicate':
            upsample_factor = int(np.prod(upsample_factor))
            # Repeat for upsampling.
            self.upsample = nn.Upsample(scale_factor=upsample_factor, mode='nearest')
        elif upsample_method == 'transposed_conv2d':
            if not isinstance(upsample_factor, list):
                raise ValueError("You must specify upsample_factor as a list "
                                 "when used with transposed_conv2d")
            freq_axis_kernel_size = 3
            self.upsample_conv = nn.ModuleList()
            for s in upsample_factor:
                freq_axis_padding = (freq_axis_kernel_size - 1) // 2
                # The filter sizes (in time) are doubled from strides
                # to avoid the checkerboard artifacts.
                conv = nn.ConvTranspose2d(1, 1, (freq_axis_kernel_size, 2 * s),
                                          padding=(freq_axis_padding, s // 2),
                                          dilation=1, stride=(1, s))
                self.upsample_conv.append(conv)
                self.upsample_conv.append(nn.LeakyReLU(negative_slope=0.4, inplace=True))
        else:
            raise ValueError('{} upsampling is not supported'.format(self._upsample_method))

        self.squeeze_layer = SqueezeLayer(squeeze_factor)

    def forward(self, input):
        if self.upsample_method == 'duplicate':
            output = self.upsample(input)
        elif self.upsample_method == 'transposed_conv2d':
            output = input.unsqueeze(1)
            for layer in self.upsample_conv:
                output = layer(output)
            output = output.squeeze(1)
        output = self.squeeze_layer(output)[0]
        return output
