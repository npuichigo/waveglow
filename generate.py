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

import argparse
import json
import math
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import waveglow.logging as logger

from scipy.io import wavfile
from torch.nn import functional as F
from waveglow.model import WaveGlow
from waveglow.modules import UpsampleNet



# Basic model parameters as external flags.
FLAGS = None


def write_wav(wav, sample_rate, filename):
    max_value_16bit = (1 << 15) - 1
    wav *= max_value_16bit
    wavfile.write(filename, sample_rate, wav.astype(np.int16))
    logger.info('Updated wav file at {}'.format(filename))


def build_model(params):
    upsample_net = UpsampleNet(
        upsample_factor=params['upsample_net']['upsample_factor'],
        upsample_method=params['upsample_net']['upsample_method'],
        squeeze_factor=params['waveglow']['squeeze_factor'])

    input_channels = params['upsample_net']['input_channels']
    local_condition_channels = input_channels * params['waveglow']['squeeze_factor']
    model = WaveGlow(
        squeeze_factor=params['waveglow']['squeeze_factor'],
        num_layers=params['waveglow']['num_layers'],
        wn_filter_width=params['waveglow']['wn_filter_width'],
        wn_dilation_layers=params['waveglow']['wn_dilation_layers'],
        wn_residual_channels=params['waveglow']['wn_residual_channels'],
        wn_dilation_channels=params['waveglow']['wn_dilation_channels'],
        wn_skip_channels=params['waveglow']['wn_skip_channels'],
        local_condition_channels=local_condition_channels)

    return upsample_net, model


def main(_):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(FLAGS.waveglow_params, 'r') as f:
        waveglow_params = json.load(f)

    upsample_net, model = build_model(waveglow_params)
    print(upsample_net)
    print(model)

    checkpoint = torch.load(FLAGS.checkpoint,
                            map_location=lambda storage, loc: storage)
    upsample_net.load_state_dict(checkpoint['upsample_net'])
    model.load_state_dict(checkpoint['waveglow'])

    upsample_net.to(device).eval()
    model.to(device).eval()

    with torch.no_grad():
        local_condition = np.load(FLAGS.local_condition_file)

        local_condition = torch.FloatTensor(local_condition).to(device)
        local_condition = local_condition.unsqueeze(0).transpose(1, 2)
        local_condition = upsample_net(local_condition)

        noise = torch.FloatTensor(
            1, waveglow_params['waveglow']['squeeze_factor'],
            local_condition.shape[2]).normal_(0.0, 0.6)
        noise, local_condition = noise.to(device), local_condition.to(device)

        logger.info("Generating samples...")
        waveform = model(noise, reverse=True, logdet=None, local_condition=local_condition)
        waveform = torch.clamp(torch.clamp(waveform[0], min=-1.), max=1.)
        waveform = waveform.cpu().numpy()

        wav_path = os.path.splitext(
            os.path.basename(FLAGS.local_condition_file))[0] + '.wav'
        wav_path = os.path.join(FLAGS.output, wav_path)
        write_wav(waveform, waveglow_params['waveglow']['sample_rate'], wav_path)


if __name__ == '__main__':
    logger.set_verbosity(logger.INFO)

    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--local_condition_file',
        type=str,
        required=True,
        help='Local condition file or folder to predict.')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Checkpoint to restore model parameters.')
    parser.add_argument(
        '--waveglow_params',
        type=str,
        default='waveglow_params.json',
        help='JSON file with the network parameters.')
    parser.add_argument(
        '--use_cuda',
        type=_str_to_bool,
        default=True,
        help='Enables CUDA training.')
    parser.add_argument(
        '--output',
        type=str,
        default='.',
        help='Output folder.')
    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.use_cuda = FLAGS.use_cuda and torch.cuda.is_available()
    main([sys.argv[0]] + unparsed)
