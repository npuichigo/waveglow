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

import fnmatch
import io
import numpy as np
import os
import soundfile as sf
import torch

from torch.utils.data import Dataset


class WaveGlowDataset(Dataset):
    """WaveGlow dataset."""

    def __init__(self,
                 audio_dir,
                 sample_rate,
                 local_condition_enabled=False,
                 local_condition_dir=None):
        """Initializes the WaveGlowDataset.

        Args:
            audio_dir: Directory for audio data.
            sample_rate: Sample rate of audio.
            local_condition_enabled: Whether to use local condition.
            local_condition_dir: Directory for local condition.
        """
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.local_condition_enabled = local_condition_enabled
        self.local_condition_dir = local_condition_dir

        self.audio_files = sorted(self._find_files(audio_dir))
        if not self.audio_files:
            raise ValueError("No audio files found in '{}'.".format(audio_dir))

        if self.local_condition_enabled:
            self.local_condition_files = sorted(self._find_files(local_condition_dir, '*.npy'))
            self._check_consistency(self.audio_files, self.local_condition_files)

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, index):
        sample, sample_rate = sf.read(self.audio_files[index])
        assert sample_rate == self.sample_rate
        sample = sample.reshape(-1, 1)
        if self.local_condition_enabled:
            local_condition = np.load(self.local_condition_files[index])
            return sample, local_condition
        else:
            return sample

    def _find_files(self, directory, pattern='*.wav'):
        """Recursively finds all files matching the pattern."""
        files = []
        for root, dirnames, filenames in os.walk(directory):
            for filename in fnmatch.filter(filenames, pattern):
                files.append(os.path.join(root, filename))
        return files

    def _check_consistency(self, audio_namelist, local_condition_namelist):
        audio_ids = [os.path.splitext(os.path.basename(name))[0]
                     for name in audio_namelist]
        local_condition_ids = [os.path.splitext(os.path.basename(name))[0]
                               for name in local_condition_namelist]
        if audio_ids != local_condition_ids:
            raise ValueError("Mismatch between audio files and local condition files.")


class WaveGlowCollate(object):
    """Function object used as a collate function for DataLoader."""

    def __init__(self, sample_size, upsample_factor, local_condition_enabled=None):
        """Initializes the WaveGlowCollate.

        Args:
            sample_size: The size of audio clips for training. 
            upsample_factor: The upsampling factor between sample and local condition.
            local_condition_enabled: Whether to use local condition.
        """
        self.sample_size = sample_size
        self.upsample_factor = int(np.prod(upsample_factor))
        assert sample_size % self.upsample_factor == 0
        self.local_condition_enabled = local_condition_enabled

    def _collate_fn(self, batch):
        if self.local_condition_enabled:
            new_batch = []
            for idx in range(len(batch)):
                sample, local_condition = batch[idx]

                # Pad utterance tail with silence, and cut tail if too much silence.
                sample_size = len(sample)
                local_condition_size = len(local_condition)
                length_diff = self.upsample_factor * local_condition_size - sample_size
                if length_diff > 0:
                    sample = np.pad(
                        sample, [[0, length_diff], [0, 0]], 'constant')
                elif length_diff < 0:
                    sample = sample[:self.upsample_factor * local_condition_size]

                if len(sample) > self.sample_size:
                    frame_size  = self.sample_size // self.upsample_factor 
                    lc_beg = np.random.randint(0, len(local_condition) - frame_size)
                    sample_beg = lc_beg * self.upsample_factor
                    sample = sample[sample_beg:sample_beg + self.sample_size, :]
                    local_condition = local_condition[lc_beg:lc_beg + frame_size, :]
                new_batch.append((sample, local_condition))

            # Dynamic padding.
            max_len = max([len(x[0]) for x in new_batch])
            sample_batch = [
                np.pad(x[0], [[0, max_len - len(x[0])], [0, 0]], 'constant')
                for x in new_batch
            ]
            max_len = max([len(x[1]) for x in new_batch])
            local_condition_batch = [
                np.pad(x[1], [[0, max_len - len(x[1])], [0, 0]], 'edge')
                for x in new_batch
            ]

            # scalar output
            sample_batch = np.array(sample_batch)
            sample_batch = torch.FloatTensor(sample_batch).transpose(1, 2)

            # Local condition should be one timestep ahead of samples.
            local_condition_batch = torch.FloatTensor(
                np.array(local_condition_batch)).transpose(1, 2)

            return sample_batch, local_condition_batch
        else:
            new_batch = []
            for idx in range(len(batch)):
                sample = batch[idx]
                if len(sample) > self.sample_size:
                    sample_beg = np.random.randint(0, len(sample) - self.sample_size)
                    sample = sample[sample_beg : sample_beg + self.sample_size, :]
                new_batch.append(sample)

            # Dynamic padding.
            max_len = max([len(x) for x in new_batch])
            sample_batch = [
                np.pad(x[0], [[0, max_len - len(x)], [0, 0]], 'constant')
                for x in new_batch
            ]

            # scalar output
            sample_batch = np.array(sample_batch)
            sample_batch = torch.FloatTensor(sample_batch).transpose(1, 2)

            return sample_batch

    def __call__(self, batch):
        return self._collate_fn(batch)
