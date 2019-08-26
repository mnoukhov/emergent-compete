import math

import gin
import torch
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss


class CirclePointsIter:
    def __init__(self, num_points, bias, batch_size, num_batches, device, training):
        self.num_points = num_points
        self.bias = bias
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.device = device
        self.training = training

        self.batches = 0

    def __next__(self):
        if self.batches >= self.num_batches:
            raise StopIteration()

        if self.training:
            send_targets = self.num_points * torch.rand(size=(self.batch_size, 1),
                                                        device=self.device)
        else:
            send_targets = torch.arange(0, self.num_points,
                                        step=self.num_points / self.batch_size,
                                        device=self.device).unsqueeze(1)

        recv_targets = (send_targets + self.bias) % self.num_points

        self.batches += 1

        return (send_targets, recv_targets)


@gin.configurable
class Game(DataLoader):
    def __init__(self, num_points, bias, batch_size, num_batches, device='cpu', training=True):
        self.batch_size = batch_size
        self.num_points = num_points
        self.bias = bias
        self.num_batches = num_batches
        self.device = device
        self.training = training

    def __iter__(self):
        return CirclePointsIter(self.num_points, self.bias, self.batch_size,
                                self.num_batches, self.device, training=self.training)


class CircleL1(_Loss):
    def __init__(self, num_points):
        super().__init__(reduction=None)
        self.num_points = num_points

    def forward(self, output, target):
        # torch.remainder has an issue
        pred = torch.abs(torch.fmod(output, self.num_points))
        diff = torch.abs(pred - target)
        counter_diff = self.num_points - diff
        min_diff = torch.min(diff, counter_diff)
        return min_diff


class CircleL2(_Loss):
    def __init__(self, num_points):
        super().__init__(reduction=None)
        self.num_points = num_points

    def forward(self, output, target):
        pred = torch.abs(torch.fmod(output, self.num_points))
        diff = torch.abs(pred - target)
        counter_diff = self.num_points - diff
        min_diff = torch.min(diff, counter_diff)
        return min_diff

