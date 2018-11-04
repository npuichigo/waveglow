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
"""Implementation of the WaveGlow model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn

from torch.nn import functional as F
from waveglow.modules import FlowStep, SqueezeLayer


class WaveGlow(nn.Module):
    """Implements the WaveGlow model."""

    def __init__(self,
                 squeeze_factor=8,
                 num_layers=12,
                 wn_filter_width=3,
                 wn_dilation_layers=8,
                 wn_residual_channels=512,
                 wn_dilation_channels=256,
                 wn_skip_channels=256,
                 local_condition_channels=None):
        """Initializes the WaveGlow model.

        Args:
            local_condition_channels: Number of channels in local conditioning
                vector. None indicates there is no local conditioning.
        """
        super(WaveGlow, self).__init__()

        self.squeeze_factor = squeeze_factor
        self.num_layers = num_layers
        self.num_scales = int(np.log2(squeeze_factor)) + 1

        self.squeeze_layer = SqueezeLayer(squeeze_factor)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                FlowStep(squeeze_factor,
                         wn_filter_width=wn_filter_width,
                         wn_dilation_layers=wn_dilation_layers,
                         wn_residual_channels=wn_residual_channels,
                         wn_dilation_channels=wn_dilation_channels,
                         wn_skip_channels=wn_skip_channels,
                         local_condition_channels=local_condition_channels))
            # multi-scale architecture
            if (i + 1) % self.num_scales == 0:
                squeeze_factor //= 2

    def forward(self, input, logdet, reverse, local_condition):
        if not reverse:
            output, logdet = self.squeeze_layer(input, logdet=logdet, rerverse=False)

            early_outputs = []
            for i, layer in enumerate(self.layers):
                output, logdet = layer(output, logdet=logdet, reverse=False,
                                       local_condition=local_condition)

                if (i + 1) % self.num_scales == 0:
                    early_output, output = output.split(output.size(1) // 2, 1)
                    early_outputs.append(early_output)
            early_outputs.append(output)

            return torch.cat(early_outputs, 1), logdet
        else:
            output = input
            for i, layer in enumerate(reversed(self.layers)):
                curr_input = output[:, -2 ** (i // self.num_scales + 1):, :]
                curr_output, logdet = layer(curr_input, logdet=logdet, reverse=True,
                                            local_condition=local_condition)
                output[:, -2 ** (i // self.num_scales + 1):, :] = curr_output

            output, logdet = self.squeeze_layer(output, logdet=logdet, reverse=True)

            return output, logdet
