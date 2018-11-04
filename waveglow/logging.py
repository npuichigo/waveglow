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

import logging as _logging
import os as _os
import sys as _sys
import time as _time
from logging import DEBUG
from logging import ERROR
from logging import FATAL
from logging import INFO
from logging import WARN
import threading


# Don't use this directly. Use _get_logger() instead.
_logger = None
_logger_lock = threading.Lock()


def _get_logger():
    global _logger

    # Use double-checked locking to avoid taking lock unnecessarily.
    if _logger:
        return _logger

    _logger_lock.acquire()

    try:
        if _logger:
            return _logger

        # Scope the pytorch logger to not conflict with users' loggers.
        logger = _logging.getLogger('pytorch')

        # Don't further configure the TensorFlow logger if the root logger is
        # already configured. This prevents double logging in those cases.
        if not _logging.getLogger().handlers:
            # Determine whether we are in an interactive environment
            _interactive = False
            try:
                # This is only defined in interactive shells.
                if _sys.ps1: _interactive = True
            except AttributeError:
                # Even now, we may be in an interactive shell with `python -i`.
                _interactive = _sys.flags.interactive

            # If we are in an interactive environment (like Jupyter), set loglevel
            # to INFO and pipe the output to stdout.
            if _interactive:
                logger.setLevel(INFO)
                _logging_target = _sys.stdout
            else:
                _logging_target = _sys.stderr

            # Add the output handler.
            _handler = _logging.StreamHandler(_logging_target)
            _handler.setFormatter(_logging.Formatter(_logging.BASIC_FORMAT, None))
            logger.addHandler(_handler)

        _logger = logger
        return _logger

    finally:
        _logger_lock.release()


def log(level, msg, *args, **kwargs):
    _get_logger().log(level, msg, *args, **kwargs)


def debug(msg, *args, **kwargs):
    _get_logger().debug(msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    _get_logger().error(msg, *args, **kwargs)


def fatal(msg, *args, **kwargs):
    _get_logger().fatal(msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    _get_logger().info(msg, *args, **kwargs)


def warn(msg, *args, **kwargs):
    _get_logger().warn(msg, *args, **kwargs)


def warning(msg, *args, **kwargs):
    _get_logger().warning(msg, *args, **kwargs)

# Counter to keep track of number of log entries per token.
_log_counter_per_token = {}


def vlog(level, msg, *args, **kwargs):
    _get_logger().log(level, msg, *args, **kwargs)


def _GetNextLogCountPerToken(token):
    """Wrapper for _log_counter_per_token.
    Args:
        token: The token for which to look up the count.
    Returns:
        The number of times this function has been called with
        *token* as an argument (starting at 0)
    """
    global _log_counter_per_token  # pylint: disable=global-variable-not-assigned
    _log_counter_per_token[token] = 1 + _log_counter_per_token.get(token, -1)
    return _log_counter_per_token[token]


def log_every_n(level, msg, n, *args):
    """Log 'msg % args' at level 'level' once per 'n' times.
    Logs the 1st call, (N+1)st call, (2N+1)st call,  etc.
    Not threadsafe.
    Args:
        level: The level at which to log.
        msg: The message to be logged.
        n: The number of times this should be called before it is logged.
        *args: The args to be substituted into the msg.
    """
    count = _GetNextLogCountPerToken(_GetFileAndLine())
    log_if(level, msg, not (count % n), *args)


def log_first_n(level, msg, n, *args):  # pylint: disable=g-bad-name
    """Log 'msg % args' at level 'level' only first 'n' times.
    Not threadsafe.
    Args:
        level: The level at which to log.
        msg: The message to be logged.
        n: The number of times this should be called before it is logged.
        *args: The args to be substituted into the msg.
    """
    count = _GetNextLogCountPerToken(_GetFileAndLine())
    log_if(level, msg, count < n, *args)


def log_if(level, msg, condition, *args):
    """Log 'msg % args' at level 'level' only if condition is fulfilled."""
    if condition:
        vlog(level, msg, *args)


def get_verbosity():
    """Return how much logging output will be produced."""
    return _get_logger().getEffectiveLevel()


def set_verbosity(v):
    """Sets the threshold for what messages will be logged."""
    _get_logger().setLevel(v)
