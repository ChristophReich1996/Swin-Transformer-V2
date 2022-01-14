# Source: https://github.com/kuangliu/pytorch-cifar/blob/master/utils.py
from typing import List

import time
import os
import sys
import torch
import torch.nn as nn

from swin_transformer_v2 import SwinTransformerV2

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 30.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


class ClassificationModelWrapper(nn.Module):
    """
    Wraps a Swin Transformer V2 model to perform image classification.
    """

    def __init__(self, model: SwinTransformerV2, number_of_classes: int = 10, output_channels: int = 768) -> None:
        """
        Constructor method
        :param model: (SwinTransformerV2) Swin Transformer V2 model
        :param number_of_classes: (int) Number of classes to predict
        :param output_channels: (int) Output channels of the last feature map of the Swin Transformer V2 model
        """
        # Call super constructor
        super(ClassificationModelWrapper, self).__init__()
        # Save model
        self.model: SwinTransformerV2 = model
        # Init adaptive average pooling layer
        self.pooling: nn.Module = nn.AdaptiveAvgPool2d(1)
        # Init classification head
        self.classification_head: nn.Module = nn.Linear(in_features=output_channels, out_features=number_of_classes)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor of the shape [batch size, channels, height, width]
        :return: (torch.Tensor) Output classification of the shape [batch size, number of classes]
        """
        # Compute features
        features: List[torch.Tensor] = self.model(input)
        # Compute classification
        classification: torch.Tensor = self.classification_head(self.pooling(features[-1]).flatten(start_dim=1))
        return classification
